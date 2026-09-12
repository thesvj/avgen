"""Tests for :mod:`avgen.rl` — the SDE policy, GRPO, DPO, and the rewards.

Reinforcement learning on a diffusion model rests on a small number of exact
algebraic identities, and every one of them is silent when it breaks: the loss
still comes out finite, the curve still moves, and the run is simply optimising
something other than what it claims to. This file asserts the identities rather
than the shapes.

**The policy exists at all only because of the SDE.** A rectified-flow sampler is
a deterministic map, so its trajectory distribution is a point mass, every
importance ratio is identically one, and the policy gradient is identically zero.
:func:`~avgen.rl.sde.to_sde` replaces it with a marginal-preserving SDE. The two
properties that must hold exactly are that zero noise recovers the ODE
*bit-for-bit* (so the deterministic path is not perturbed by being expressed as a
limit) and that replaying a stored sample under the same velocity recovers the
same log-probability *exactly* (so ``log pi_theta - log pi_old`` is zero on the
first inner pass rather than merely small).

**The first inner pass must be a no-op.** Nothing has changed yet, so the ratio
is exactly 1.0 and the clip fraction is exactly 0.0. That is a sharp assertion:
any leakage — a re-sampled transition instead of a replayed one, a mean computed
from a different velocity, an advantage attached to the wrong step — moves it off
1.0 immediately.

Everything runs on CPU against the real :class:`~avgen.models.VideoDiT` at the
``"tiny"`` preset for the end-to-end paths, and against pure tensors for the
algebra.
"""

from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from avgen.core import (
    GridPatchifier,
    MediaBatch,
    MediaBatchSpec,
    ModelInput,
    RNGStreams,
    TextContext,
    TokenStream,
    TrainState,
    unpatchify_grid,
)
from avgen.models import VideoDiT, preset
from avgen.rl import (
    AVSyncProxyReward,
    CompositeReward,
    DPOConfig,
    DPOObjective,
    GRPOConfig,
    GRPOTrainer,
    HPSv2Reward,
    MotionMagnitudeReward,
    RolloutBuffer,
    RunningNormalizer,
    TemporalConsistencyReward,
    WindowSchedule,
    build_reward,
    clipped_surrogate,
    diffusion_coefficient,
    distribute_prompts,
    dpo_loss,
    gaussian_log_prob,
    generate_group,
    group_advantages,
    kl_penalty,
    list_rewards,
    media_tensors,
    model_velocity_fn,
    ode_step,
    regulated_log_ratio,
    rollout_velocity_batch,
    to_sde,
)
from avgen.train.objective import Objective, ObjectiveOutput

FRAMES = 4
EXTENT = 4
TEXT_TOKENS = 6

# Rewards whose weights are a real optional dependency. They are registered so a
# config can name them and so list_rewards() tells the truth, but constructing
# one raises with the install command rather than importing at module scope.
GATED_REWARDS = frozenset({"hps_v2", "pick_score", "video_score"})

# The rewards this package computes itself, with no optional dependency and no
# downloaded weights. Named rather than derived from the registry, which also
# surfaces third-party plugins.
SHIPPED_REWARDS = frozenset(
    {"temporal_consistency", "motion_magnitude", "av_sync_proxy"}
)


def build_model(seed: int = 0) -> VideoDiT:
    """Return a tiny VideoDiT whose output is not identically zero.

    ``init_weights`` zero-initialises ``final_proj``, so a fresh model predicts
    exactly zero and every trajectory would be a straight line. The output head
    is given real weights so the rollouts carry actual structure.
    """
    torch.manual_seed(seed)
    model = VideoDiT(preset("tiny"))
    model.init_weights()
    with torch.no_grad():
        nn.init.normal_(model.final_proj.weight, std=0.02)
    return model


def make_velocity_fn(model: VideoDiT, prompt_features: torch.Tensor) -> Any:
    """Build a prompt-index-aware velocity function over the real model.

    ``prompt_index`` is honoured rather than ignored because
    :meth:`RolloutBuffer.iter_minibatches` shuffles across the (trajectory, step)
    grid: row ``i`` of an update minibatch is not row ``i`` of the rollout, so
    the conditioning has to be gathered by index or it is silently misaligned.
    """
    patchifier = model.patchifier

    def velocity(
        sample: torch.Tensor, sigma: torch.Tensor, *, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        count = sample.shape[0]
        positions = torch.arange(FRAMES, dtype=torch.float32).repeat(count, 1) / 8.0
        mask = torch.ones((count, FRAMES, EXTENT, EXTENT), dtype=torch.bool)
        stream = patchifier.to_tokens(
            sample, positions=positions, mask=mask, noise_level=sigma
        )
        inputs = ModelInput(
            video=stream,
            audio=TokenStream.empty_like(count, stream.width),
            text=TextContext(
                features=prompt_features[prompt_index],
                mask=torch.ones((count, TEXT_TOKENS), dtype=torch.bool),
            ),
        )
        return unpatchify_grid(model(inputs).video, stream.layout)

    return velocity


def make_media_batch(video: torch.Tensor, text_width: int) -> MediaBatch:
    """Build a valid audio-free MediaBatch around a video tensor."""
    total, _, frames, height, width = video.shape
    pairs = total // 2
    spec = MediaBatchSpec(
        schema_version=1,
        bucket_id=0,
        video_shape=tuple(video.shape),
        audio_shape=(total, 1, 0),
        text_shape=(total, TEXT_TOKENS, text_width),
        video_timebase_num=8,
        video_timebase_den=1,
        audio_timebase_num=1,
        audio_timebase_den=1,
        video_codec_id="test-video",
        audio_codec_id="test-audio",
    )
    # The same prompt for the winner and its loser: a preference pair is two
    # samples of one prompt, and a different prompt on each side would make the
    # difference of errors reflect the prompts rather than the preference.
    text = torch.randn((pairs, TEXT_TOKENS, text_width)).repeat(2, 1, 1)
    return MediaBatch(
        video=video,
        audio=torch.zeros((total, 1, 0)),
        text=text,
        video_mask=torch.ones((total, frames, height, width), dtype=torch.bool),
        audio_mask=torch.zeros((total, 0), dtype=torch.bool),
        video_positions=torch.arange(frames, dtype=torch.float32).repeat(total, 1)
        / 8.0,
        audio_positions=torch.zeros((total, 0)),
        sample_ids=torch.arange(total, dtype=torch.int64),
        spec=spec,
    )


class TestSDEReducesToTheODE:
    """Zero noise must recover the deterministic sampler exactly.

    This is not a convenience. MixGRPO runs every step outside its window as a
    plain ODE step and every step inside it as an SDE step, on the same
    trajectory. If the two paths disagreed even at the level of rounding, the
    stored transitions would describe states the deterministic segments never
    visited, and the ratio would be measuring discretisation noise rather than a
    policy change.

    The zero-noise branch is taken exactly rather than as a limit of the
    Gaussian: a Dirac transition has infinite log-density, and the honest value
    for the ratio it induces is 1 — a log-probability of 0 contributing no policy
    gradient. That *is* the reason a deterministic sampler cannot be improved by
    policy gradient, which is what this module exists to fix.
    """

    def test_zero_noise_is_bit_identical_to_ode_step(self) -> None:
        state = torch.randn((3, 4, 4, 4, 4))
        velocity = torch.randn_like(state)
        sigma = torch.full((3,), 0.8)
        sigma_next = torch.full((3,), 0.6)
        step = to_sde(velocity, state, sigma, sigma_next, noise_level=0.0)
        assert torch.equal(step.sample, ode_step(velocity, state, sigma, sigma_next))

    def test_zero_noise_reports_itself_deterministic(self) -> None:
        state = torch.randn((2, 4, 2, 2, 2))
        velocity = torch.randn_like(state)
        step = to_sde(
            velocity,
            state,
            torch.full((2,), 0.5),
            torch.full((2,), 0.25),
            noise_level=0.0,
        )
        assert step.deterministic
        # Zero by convention rather than infinite, so the ratio it induces is 1.
        assert bool((step.log_prob == 0.0).all())
        assert bool((step.std == 0.0).all())

    def test_zero_noise_scores_a_stored_sample_without_resampling(self) -> None:
        state = torch.randn((2, 4, 2, 2, 2))
        velocity = torch.randn_like(state)
        stored = torch.randn_like(state)
        step = to_sde(
            velocity,
            state,
            torch.full((2,), 0.5),
            torch.full((2,), 0.25),
            noise_level=0.0,
            prev_sample=stored,
        )
        assert torch.equal(step.sample, stored)

    def test_a_nonzero_noise_step_departs_from_the_ode(self) -> None:
        # The control: if the stochastic path also equalled the ODE, the policy
        # would still have zero entropy and the identity above would prove
        # nothing.
        state = torch.randn((3, 4, 4, 4, 4))
        velocity = torch.randn_like(state)
        sigma = torch.full((3,), 0.8)
        sigma_next = torch.full((3,), 0.6)
        generator = torch.Generator().manual_seed(7)
        step = to_sde(
            velocity, state, sigma, sigma_next, noise_level=0.7, generator=generator
        )
        assert not step.deterministic
        assert bool((step.std > 0.0).all())
        assert not torch.allclose(
            step.sample, ode_step(velocity, state, sigma, sigma_next)
        )


class TestTransitionLogProbs:
    """A replayed log-probability must equal the stored one exactly.

    The policy ratio is ``exp(log pi_theta - log pi_old)``, and on the first
    inner pass ``pi_theta`` *is* ``pi_old``. If the replay disagreed with the
    rollout — because it re-sampled, or rebuilt the mean differently — the ratio
    would start off 1 and the clip range, which is 2e-6 for a regulated ratio,
    would bind on noise.
    """

    def test_replay_recovers_the_rollout_log_prob_exactly(self) -> None:
        state = torch.randn((3, 4, 4, 4, 4))
        velocity = torch.randn_like(state)
        sigma = torch.full((3,), 0.8)
        sigma_next = torch.full((3,), 0.6)
        generator = torch.Generator().manual_seed(11)
        rollout = to_sde(
            velocity, state, sigma, sigma_next, noise_level=0.7, generator=generator
        )
        replay = to_sde(
            velocity,
            state,
            sigma,
            sigma_next,
            noise_level=0.7,
            prev_sample=rollout.sample,
        )
        assert torch.equal(replay.log_prob, rollout.log_prob)
        assert torch.equal(replay.mean, rollout.mean)
        assert torch.equal(replay.sample, rollout.sample)

    def test_log_probs_are_finite_across_the_schedule(self) -> None:
        # The flow_grpo coefficient is a*sqrt(s/(1-s)), which diverges at s = 1;
        # the floor is what keeps the first rollout step finite.
        state = torch.randn((4, 4, 2, 2, 2))
        velocity = torch.randn_like(state)
        generator = torch.Generator().manual_seed(3)
        levels = torch.linspace(1.0, 0.0, 9)
        for index in range(len(levels) - 1):
            step = to_sde(
                velocity,
                state,
                levels[index].expand(4).contiguous(),
                levels[index + 1].expand(4).contiguous(),
                noise_level=0.7,
                generator=generator,
            )
            assert bool(torch.isfinite(step.log_prob).all())
            assert bool(torch.isfinite(step.sample).all())

    def test_a_different_velocity_moves_the_log_prob(self) -> None:
        state = torch.randn((3, 4, 2, 2, 2))
        velocity = torch.randn_like(state)
        sigma = torch.full((3,), 0.7)
        sigma_next = torch.full((3,), 0.5)
        generator = torch.Generator().manual_seed(5)
        rollout = to_sde(
            velocity, state, sigma, sigma_next, noise_level=0.7, generator=generator
        )
        moved = to_sde(
            velocity + 0.5,
            state,
            sigma,
            sigma_next,
            noise_level=0.7,
            prev_sample=rollout.sample,
        )
        assert not torch.allclose(moved.log_prob, rollout.log_prob)

    def test_gaussian_log_prob_matches_the_closed_form(self) -> None:
        value = torch.zeros((2, 3))
        mean = torch.zeros((2, 3))
        std = torch.ones(2)
        expected = -0.5 * math.log(2.0 * math.pi) * 3.0
        torch.testing.assert_close(
            gaussian_log_prob(value, mean, std),
            torch.full((2,), expected),
        )

    def test_a_mask_excludes_padding_from_the_log_prob(self) -> None:
        # Without the mask the log-probability grows with the padding, and two
        # samples in different buckets stop being comparable.
        value = torch.randn((2, 6))
        mean = torch.zeros((2, 6))
        std = torch.ones(2)
        mask = torch.zeros((2, 6), dtype=torch.bool)
        mask[:, :3] = True
        masked = gaussian_log_prob(value, mean, std, mask=mask)
        halved = gaussian_log_prob(value[:, :3], mean[:, :3], std)
        torch.testing.assert_close(masked, halved)

    def test_a_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must match mean shape"):
            gaussian_log_prob(torch.zeros((2, 3)), torch.zeros((2, 4)), torch.ones(2))

    def test_the_noise_level_must_fall(self) -> None:
        # Sampling integrates the noise level downward toward clean; a rising
        # schedule would give a negative variance under the square root.
        state = torch.randn((2, 4))
        with pytest.raises(ValueError, match="must not exceed sigma_t"):
            to_sde(
                torch.randn_like(state),
                state,
                torch.full((2,), 0.3),
                torch.full((2,), 0.6),
            )

    def test_a_mismatched_prev_sample_raises(self) -> None:
        state = torch.randn((2, 4))
        with pytest.raises(ValueError, match="must match x_t"):
            to_sde(
                torch.randn_like(state),
                state,
                torch.full((2,), 0.6),
                torch.full((2,), 0.3),
                prev_sample=torch.randn((2, 5)),
            )


class TestDiffusionCoefficient:
    """The diffusion coefficient is the entropy knob, and zero must mean zero."""

    def test_constant_schedule_is_the_scale(self) -> None:
        value = diffusion_coefficient(
            torch.tensor([0.1, 0.9]), noise_level=0.3, schedule="constant"
        )
        torch.testing.assert_close(value, torch.full((2,), 0.3))

    def test_flow_grpo_schedule_grows_with_noise(self) -> None:
        value = diffusion_coefficient(torch.tensor([0.1, 0.5, 0.9]), noise_level=0.7)
        assert bool((value[1:] > value[:-1]).all())

    def test_the_floor_keeps_the_endpoint_finite(self) -> None:
        # At s = 1 the coefficient is infinite and the first rollout step would
        # destroy the state.
        value = diffusion_coefficient(torch.tensor([1.0]), noise_level=0.7)
        assert bool(torch.isfinite(value).all())

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"noise_level": -1.0}, "non-negative"),
            ({"noise_level": 0.7, "floor": 0.0}, "floor must be"),
            ({"noise_level": 0.7, "schedule": "cosine"}, "unknown schedule"),
        ],
    )
    def test_invalid_arguments_raise(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            diffusion_coefficient(torch.tensor([0.5]), **kwargs)


class TestGroupAdvantages:
    """The group mean is the entire baseline, so the grouping must be exact.

    GRPO has no value network on purpose: a critic for a video diffusion policy
    would be a second video-sized model trained on one scalar per trajectory, and
    its bias would be indistinguishable in the loss from the policy's own error.
    The price is that a mis-grouped batch computes each advantage against the
    wrong baseline, and nothing in the loss curve reveals it.
    """

    def test_each_group_has_zero_mean(self) -> None:
        rewards = torch.randn(12)
        advantages = group_advantages(rewards, 4)
        means = advantages.reshape(-1, 4).mean(dim=1)
        assert bool(means.abs().max() < 1e-5)

    def test_std_normalisation_makes_one_clip_range_work_everywhere(self) -> None:
        # A reward on the scale of 1 and one on the scale of 40 must produce the
        # same advantage distribution, which is what lets one clip range serve
        # both. (The epsilon on the standard deviation makes this only
        # approximate once the reward scale approaches 1e-4 itself.)
        small = group_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]), 4)
        large = group_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]) * 40.0, 4)
        torch.testing.assert_close(small, large, rtol=1e-3, atol=1e-3)

    def test_without_std_normalisation_the_scale_survives(self) -> None:
        # Dr. GRPO's correction: not dividing removes a bias that favours prompts
        # the policy already answers consistently.
        raw = group_advantages(
            torch.tensor([1.0, 2.0, 3.0, 4.0]), 4, normalize_by_std=False
        )
        torch.testing.assert_close(raw, torch.tensor([-1.5, -0.5, 0.5, 1.5]))

    def test_a_group_of_identical_rewards_gives_zero_advantage(self) -> None:
        # The documented no-op case, and the correct one: every sample in the
        # group was equally good, so there is nothing to reinforce. The epsilon
        # on the standard deviation is what keeps 0/0 out of the result.
        advantages = group_advantages(torch.full((4,), 3.0), 4)
        assert torch.equal(advantages, torch.zeros(4))

    def test_the_clamp_bounds_an_extreme_advantage(self) -> None:
        # An unclipped advantage can be arbitrarily large and will dominate a
        # whole batch. Shown on the un-normalised path, because with std
        # normalisation a group of G has a standardised range of only
        # sqrt(G - 1) and a clip of 5 can never bind at G = 4.
        rewards = torch.tensor([0.0, 0.0, 0.0, 100.0])
        clipped = group_advantages(rewards, 4, normalize_by_std=False, clip=5.0)
        loose = group_advantages(rewards, 4, normalize_by_std=False, clip=None)
        assert float(clipped.abs().max()) == pytest.approx(5.0)
        assert float(loose.abs().max()) == pytest.approx(75.0)

    def test_groups_are_contiguous_not_interleaved(self) -> None:
        # reshape(-1, group_size) groups adjacent samples, which is why
        # generate_group uses repeat_interleave and not repeat.
        rewards = torch.tensor([0.0, 1.0, 10.0, 11.0])
        advantages = group_advantages(rewards, 2, normalize_by_std=False)
        torch.testing.assert_close(advantages, torch.tensor([-0.5, 0.5, -0.5, 0.5]))

    def test_a_group_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            group_advantages(torch.randn(4), 1)

    def test_a_ragged_group_raises(self) -> None:
        with pytest.raises(ValueError, match="not a multiple"):
            group_advantages(torch.randn(5), 2)


class TestSurrogateObjective:
    """On the first inner pass nothing has changed, so the ratio is exactly 1.

    That is the sharpest available check on the whole update path. The regulated
    log ratio is ``-(dmu . eps)``: when the current policy's mean equals the
    rollout policy's mean, ``dmu`` is exactly zero, the sum is exactly zero, and
    ``exp(0)`` is exactly ``1.0`` — with no clipped samples. Any drift means the
    update is scoring a transition the rollout did not take.
    """

    def test_an_unchanged_policy_gives_ratio_one_and_no_clipping(self) -> None:
        mean = torch.randn((5, 3, 4))
        sample = torch.randn((5, 3, 4))
        std = torch.rand(5) + 0.5
        log_ratio = regulated_log_ratio(mean, mean, std, sample)
        assert torch.equal(log_ratio, torch.zeros(5))
        loss, clip_fraction, ratio = clipped_surrogate(
            log_ratio, torch.randn(5), clip_low=2e-6, clip_high=2e-6
        )
        assert torch.equal(ratio, torch.ones(5))
        assert float(clip_fraction) == 0.0
        assert bool(torch.isfinite(loss))

    def test_the_regulated_ratio_is_the_published_identity(self) -> None:
        # GRPO-Guard Eq. 8: multiplying the raw log ratio by s and adding back
        # the deterministic bias leaves -(dmu . eps), which has the same
        # distribution at every timestep. Asserted against the definition so a
        # refactor cannot quietly change which quantity is being clipped.
        mean_old = torch.randn((4, 6))
        mean_new = torch.randn((4, 6))
        std = torch.rand(4) + 0.5
        epsilon = torch.randn((4, 6))
        sample = mean_old + std.unsqueeze(-1) * epsilon
        expected = -((mean_old - mean_new) * epsilon).sum(dim=1)
        torch.testing.assert_close(
            regulated_log_ratio(mean_new, mean_old, std, sample), expected
        )

    def test_a_mask_excludes_padding_from_the_ratio(self) -> None:
        mean_old = torch.randn((2, 8))
        mean_new = torch.randn((2, 8))
        std = torch.ones(2)
        sample = torch.randn((2, 8))
        mask = torch.zeros((2, 8), dtype=torch.bool)
        mask[:, :4] = True
        masked = regulated_log_ratio(mean_new, mean_old, std, sample, mask=mask)
        partial = regulated_log_ratio(
            mean_new[:, :4], mean_old[:, :4], std, sample[:, :4]
        )
        torch.testing.assert_close(masked, partial)

    def test_the_clip_binds_once_the_ratio_moves(self) -> None:
        log_ratio = torch.tensor([0.0, 0.5, -0.5])
        _, clip_fraction, ratio = clipped_surrogate(
            log_ratio, torch.ones(3), clip_low=0.1, clip_high=0.1
        )
        assert float(clip_fraction) == pytest.approx(2.0 / 3.0)
        assert float(ratio[0]) == 1.0

    def test_the_objective_is_pessimistic(self) -> None:
        # min of the clipped and unclipped terms is a lower bound on the true
        # objective, so an update that looks good only because the ratio drifted
        # is not credited. With a positive advantage and a ratio above the
        # ceiling, the clipped term wins and the loss stops improving.
        advantage = torch.ones(1)
        loose, _, _ = clipped_surrogate(
            torch.tensor([1.0]), advantage, clip_low=0.1, clip_high=0.1
        )
        tight, _, _ = clipped_surrogate(
            torch.tensor([2.0]), advantage, clip_low=0.1, clip_high=0.1
        )
        assert float(loose) == pytest.approx(float(tight))

    def test_weights_reweight_the_mean(self) -> None:
        log_ratio = torch.zeros(4)
        advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
        weights = torch.tensor([2.0, 2.0, 0.0, 0.0])
        loss, _, _ = clipped_surrogate(
            log_ratio, advantage, clip_low=0.1, clip_high=0.1, weights=weights
        )
        assert float(loss) == pytest.approx(-1.0)


class TestKLPenalty:
    """Schulman's k3 estimator must be non-negative per sample, not in expectation.

    The penalty is added to a loss. A per-sample-negative KL estimate would
    occasionally *pay* the policy for moving away from the reference, and the
    naive ``log pi_new - log pi_ref`` estimator has enough variance for that to
    happen often.
    """

    def test_an_identical_reference_gives_exactly_zero(self) -> None:
        log_prob = torch.randn(8)
        assert torch.equal(kl_penalty(log_prob, log_prob), torch.zeros(8))

    def test_a_perturbed_reference_is_strictly_positive(self) -> None:
        log_prob = torch.randn(8)
        for offset in (-1.5, -0.25, 0.25, 1.5):
            penalty = kl_penalty(log_prob, log_prob + offset)
            assert bool((penalty > 0.0).all()), offset

    def test_the_estimator_is_never_negative(self) -> None:
        generator = torch.Generator().manual_seed(2)
        new = torch.randn(4096, generator=generator)
        reference = torch.randn(4096, generator=generator)
        assert bool((kl_penalty(new, reference) >= 0.0).all())


class TestWindowSchedule:
    """MixGRPO's window must sweep the trajectory and then stop at the end.

    Only steps inside the window use the SDE path, are stored, and receive
    gradient; everything else is one forward pass. That is a ~6x reduction in
    cost and memory at ``T=25, w=4``, and it costs no quality *only because the
    window moves* — over a run every timestep spends time inside it. A window
    that stopped moving, or that ran off the end of the trajectory, would leave
    part of the trajectory permanently unoptimised.
    """

    def test_the_window_starts_at_the_high_noise_end(self) -> None:
        # Deliberate: early steps decide global structure, which is what a
        # reward model responds to most strongly.
        schedule = WindowSchedule(window=4, stride=1, interval=25, total_steps=25)
        assert schedule.steps_for(0) == (0, 1, 2, 3)

    def test_the_window_slides_on_the_interval(self) -> None:
        schedule = WindowSchedule(window=2, stride=1, interval=3, total_steps=6)
        assert schedule.steps_for(0) == (0, 1)
        assert schedule.steps_for(2) == (0, 1)
        assert schedule.steps_for(3) == (1, 2)
        assert schedule.steps_for(6) == (2, 3)

    def test_the_window_clamps_at_the_clean_end(self) -> None:
        # Without the clamp the window would address step indices the rollout
        # never produced.
        schedule = WindowSchedule(window=2, stride=1, interval=1, total_steps=4)
        assert schedule.steps_for(10**6) == (2, 3)
        assert schedule.start_for(10**6) == 4 - 2

    def test_stride_moves_further_per_step(self) -> None:
        schedule = WindowSchedule(window=2, stride=2, interval=1, total_steps=8)
        assert schedule.steps_for(1) == (2, 3)
        assert schedule.steps_for(2) == (4, 5)

    def test_the_exponential_variant_sweeps_faster(self) -> None:
        # MixGRPO-Flash: a geometric interval reaches the end of the trajectory
        # in bounded time however long the run is.
        plain = WindowSchedule(window=2, stride=1, interval=3, total_steps=8)
        flash = WindowSchedule(
            window=2, stride=1, interval=3, total_steps=8, exponential=True
        )
        assert flash.start_for(12) > plain.start_for(12)
        assert flash.start_for(10**6) == 8 - 2

    def test_a_negative_step_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            WindowSchedule().start_for(-1)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window": 0}, "window must be"),
            ({"stride": 0}, "stride must be"),
            ({"interval": 0}, "interval must be"),
            ({"window": 26}, "exceeds total_steps"),
            ({"decay": 0.0}, "decay must be"),
        ],
    )
    def test_invalid_fields_raise(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            WindowSchedule(**kwargs)


class TestRolloutBuffer:
    """Stored transitions must be detached, grouped, and shuffled across steps.

    Two properties matter. Storing an attached tensor would retain the whole
    rollout's autograd graph, which for a 25-step video trajectory is the fastest
    way to run out of memory. And flattening across *steps* as well as
    trajectories is what keeps a minibatch from being one trajectory's worth of
    perfectly correlated samples wearing a batch size as a disguise.
    """

    def test_stored_transitions_carry_no_graph(self) -> None:
        buffer = RolloutBuffer(group_size=2)
        state = torch.randn((4, 3), requires_grad=True)
        step = to_sde(
            torch.randn((4, 3)),
            state,
            torch.full((4,), 0.6),
            torch.full((4,), 0.4),
            noise_level=0.7,
            generator=torch.Generator().manual_seed(1),
        )
        buffer.add_step(
            step_index=0,
            sample_before=state,
            step=step,
            sigma=torch.full((4,), 0.6),
            sigma_next=torch.full((4,), 0.4),
        )
        buffer.set_rewards(torch.randn(4), media=None)
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(4, shuffle=False))
        assert not batch.sample_before.requires_grad
        assert not batch.sample_after.requires_grad

    def test_minibatches_cover_every_transition_once(self) -> None:
        buffer = RolloutBuffer(group_size=2)
        for index in range(3):
            state = torch.randn((4, 3))
            step = to_sde(
                torch.randn((4, 3)),
                state,
                torch.full((4,), 0.6),
                torch.full((4,), 0.4),
                noise_level=0.7,
                generator=torch.Generator().manual_seed(index),
            )
            buffer.add_step(
                step_index=index,
                sample_before=state,
                step=step,
                sigma=torch.full((4,), 0.6),
                sigma_next=torch.full((4,), 0.4),
            )
        buffer.set_rewards(torch.arange(4, dtype=torch.float32), media=None)
        buffer.compute_advantages()
        assert len(buffer) == 3
        assert buffer.num_samples == 4
        sizes = [batch.size for batch in buffer.iter_minibatches(5, shuffle=False)]
        assert sum(sizes) == 12  # 3 steps x 4 trajectories

    def test_the_prompt_index_follows_the_grouping(self) -> None:
        buffer = RolloutBuffer(group_size=2)
        state = torch.randn((4, 3))
        step = to_sde(
            torch.randn((4, 3)),
            state,
            torch.full((4,), 0.6),
            torch.full((4,), 0.4),
            noise_level=0.7,
            generator=torch.Generator().manual_seed(1),
        )
        buffer.add_step(
            step_index=0,
            sample_before=state,
            step=step,
            sigma=torch.full((4,), 0.6),
            sigma_next=torch.full((4,), 0.4),
        )
        buffer.set_rewards(torch.randn(4), media=None)
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(4, shuffle=False))
        torch.testing.assert_close(batch.prompt_index, torch.tensor([0, 0, 1, 1]))

    def test_the_advantage_is_constant_along_a_trajectory(self) -> None:
        # GRPO assigns the whole trajectory's advantage to every step: the
        # reward is only observed at the end and there is no value function to
        # bootstrap an intermediate credit assignment from.
        buffer = RolloutBuffer(group_size=2)
        for index in range(2):
            state = torch.randn((2, 3))
            step = to_sde(
                torch.randn((2, 3)),
                state,
                torch.full((2,), 0.6),
                torch.full((2,), 0.4),
                noise_level=0.7,
                generator=torch.Generator().manual_seed(index),
            )
            buffer.add_step(
                step_index=index,
                sample_before=state,
                step=step,
                sigma=torch.full((2,), 0.6),
                sigma_next=torch.full((2,), 0.4),
            )
        buffer.set_rewards(torch.tensor([0.0, 1.0]), media=None)
        advantages = buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(4, shuffle=False))
        torch.testing.assert_close(batch.advantage, advantages.repeat(2))

    def test_ordering_errors_raise(self) -> None:
        buffer = RolloutBuffer(group_size=2)
        with pytest.raises(RuntimeError, match="set_rewards"):
            buffer.compute_advantages()
        state = torch.randn((2, 3))
        step = to_sde(
            torch.randn((2, 3)),
            state,
            torch.full((2,), 0.6),
            torch.full((2,), 0.4),
            noise_level=0.7,
            generator=torch.Generator().manual_seed(1),
        )
        buffer.add_step(
            step_index=0,
            sample_before=state,
            step=step,
            sigma=torch.full((2,), 0.6),
            sigma_next=torch.full((2,), 0.4),
        )
        with pytest.raises(ValueError, match="to match the trajectories"):
            buffer.set_rewards(torch.randn(3), media=None)
        buffer.set_rewards(torch.randn(2), media=None)
        with pytest.raises(RuntimeError, match="compute_advantages"):
            next(buffer.iter_minibatches(2))
        buffer.compute_advantages()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            next(buffer.iter_minibatches(0))

    def test_a_group_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            RolloutBuffer(group_size=1)


class TestGenerateGroup:
    """A group must share one initial noise draw and store only the window.

    Sharing the seed across a group removes seed variance from the comparison
    between two policy versions, which is otherwise the largest term in it. The
    ordering — ``repeat_interleave`` rather than ``repeat`` — is what makes
    ``reshape(-1, group_size)`` in ``group_advantages`` group the right samples;
    a plain repeat would interleave prompts and compute every advantage against
    the wrong baseline.
    """

    def _velocity(
        self, sample: torch.Tensor, sigma: torch.Tensor, *, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        del sigma, prompt_index
        return sample * 0.5

    def test_only_the_gradient_steps_are_stored(self) -> None:
        buffer = generate_group(
            self._velocity,
            prompts=["a", "b"],
            group_size=2,
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(2, 2),
            gradient_steps=(1, 2),
            generator=torch.Generator().manual_seed(0),
        )
        # Four denoising steps, two of them inside the window.
        assert len(buffer) == 2
        assert buffer.num_samples == 4
        assert buffer.media is not None

    def test_every_step_is_stored_by_default(self) -> None:
        buffer = generate_group(
            self._velocity,
            prompts=["a"],
            group_size=2,
            sigmas=torch.linspace(1.0, 0.0, 4),
            latent_shape=(2, 2),
            generator=torch.Generator().manual_seed(0),
        )
        assert len(buffer) == 3

    def test_supplied_initial_latents_are_shared_within_a_group(self) -> None:
        latents = torch.randn((2, 2, 2))
        buffer = generate_group(
            self._velocity,
            prompts=["a", "b"],
            group_size=3,
            sigmas=torch.tensor([1.0, 0.5]),
            latent_shape=(2, 2),
            initial_latents=latents,
            noise_level=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        buffer.set_rewards(torch.zeros(6), media=None)
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(6, shuffle=False))
        # repeat_interleave: group members are adjacent, so the first three rows
        # are prompt 0 and the last three are prompt 1.
        torch.testing.assert_close(
            batch.sample_before, latents.repeat_interleave(3, dim=0)
        )

    def test_a_rising_or_short_schedule_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two entries"):
            generate_group(
                self._velocity,
                prompts=["a"],
                group_size=2,
                sigmas=torch.tensor([1.0]),
                latent_shape=(2,),
            )
        with pytest.raises(ValueError, match="non-increasing"):
            generate_group(
                self._velocity,
                prompts=["a"],
                group_size=2,
                sigmas=torch.tensor([0.0, 1.0]),
                latent_shape=(2,),
            )

    def test_mis_shaped_initial_latents_raise(self) -> None:
        with pytest.raises(ValueError, match="initial_latents must be"):
            generate_group(
                self._velocity,
                prompts=["a", "b"],
                group_size=2,
                sigmas=torch.tensor([1.0, 0.0]),
                latent_shape=(2, 2),
                initial_latents=torch.randn((3, 2, 2)),
            )


class TestPromptDistribution:
    """A group must never span data-parallel ranks.

    A split group turns every advantage computation into a collective and makes
    the result depend on the world size, so the same run at 8 ranks and at 64
    would optimise different objectives.
    """

    def test_prompts_are_split_whole(self) -> None:
        prompts = ["a", "b", "c", "d", "e"]
        local, indices = distribute_prompts(prompts, data_rank=1, data_world=2)
        assert local == ("b", "d")
        assert indices == (1, 3)

    def test_every_prompt_reaches_exactly_one_rank(self) -> None:
        prompts = [str(i) for i in range(7)]
        seen: list[str] = []
        for rank in range(3):
            local, _ = distribute_prompts(prompts, data_rank=rank, data_world=3)
            seen.extend(local)
        assert sorted(seen) == sorted(prompts)

    @pytest.mark.parametrize(("rank", "world"), [(-1, 2), (2, 2), (0, 0)])
    def test_invalid_coordinates_raise(self, rank: int, world: int) -> None:
        with pytest.raises(ValueError, match="invalid data coordinates"):
            distribute_prompts(["a"], data_rank=rank, data_world=world)


class TestRewards:
    """Every dependency-free reward must score finite on arbitrary media.

    A reward that returns ``nan`` on one sample poisons that sample's whole
    group: the group mean and standard deviation both go non-finite, so every
    advantage in the group does, and one bad clip silently zeroes the gradient
    from ``G`` trajectories.
    """

    def test_every_dependency_free_reward_scores_finite(self) -> None:
        generator = torch.Generator().manual_seed(4)
        media = torch.randn((4, 3, 6, 8, 8), generator=generator)
        prompts = ["a prompt"] * 4
        # Named rather than counted. `list_rewards()` also surfaces anything
        # installed through the `avgen.rewards` entry-point group, so a count
        # asserts something about the developer's environment instead of about
        # the rewards this package ships.
        for name in SHIPPED_REWARDS:
            reward = build_reward(name)
            scores = reward.score(media, prompts)
            assert scores.shape == (4,), name
            assert scores.dtype is torch.float32, name
            assert bool(torch.isfinite(scores).all()), name

    def test_the_shipped_rewards_are_all_registered(self) -> None:
        """Guards the list above: a typo would make the loop test nothing."""
        registered = set(list_rewards())
        assert registered >= SHIPPED_REWARDS
        assert registered >= GATED_REWARDS

    def test_hpsv2_is_not_offered_as_an_avgen_extra(self) -> None:
        """hpsv2 pins pytest==7.2.0.

        Declaring it anywhere — even in an extra nothing else references — makes
        the universal resolution `uv lock` performs unsatisfiable, and takes
        every development environment with it. So its message must send the user
        to a standalone install, and pyproject must not name it.
        """
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        declaration = pyproject.split("[project.optional-dependencies]")[1].split(
            "[project.scripts]"
        )[0]
        assert '"hpsv2' not in declaration

        with pytest.raises(RuntimeError, match=r"pip install hpsv2") as caught:
            build_reward("hps_v2")
        assert "cannot depend on" in str(caught.value)

    def test_a_gated_reward_names_the_extra_that_provides_it(self) -> None:
        # Failing at construction rather than at import is what keeps
        # ``import avgen.rl`` working on a bare CPU box. hps_v2 is excluded
        # because it cannot be an extra at all; see the test above.
        for name in sorted(GATED_REWARDS - {"hps_v2"}):
            with pytest.raises(RuntimeError, match=r"avgen\[rewards\]") as info:
                build_reward(name)
            assert name in str(info.value)

    def test_every_gated_reward_offers_a_command_that_would_install_it(self) -> None:
        """These exist only to fail well; the message is the whole experience."""
        for name in sorted(GATED_REWARDS):
            with pytest.raises(RuntimeError, match="pip install") as info:
                build_reward(name)
            assert name in str(info.value)

    def test_the_gated_classes_raise_when_constructed_directly(self) -> None:
        with pytest.raises(RuntimeError, match="hpsv2"):
            HPSv2Reward()

    def test_an_unknown_reward_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown reward"):
            build_reward("clip_score")

    def test_temporal_consistency_peaks_on_a_still_clip(self) -> None:
        # It measures the right thing and is maximised by the wrong thing, which
        # is exactly why it belongs opposite a motion term and never alone.
        frame = torch.randn((1, 3, 1, 4, 4))
        still = frame.repeat(1, 1, 5, 1, 1)
        reward = TemporalConsistencyReward()
        assert float(reward.score(still, ["a"])) == pytest.approx(1.0, abs=1e-5)
        moving = torch.randn((1, 3, 5, 4, 4))
        assert float(reward.score(moving, ["a"])) < 1.0

    def test_a_single_frame_clip_scores_one(self) -> None:
        reward = TemporalConsistencyReward()
        assert float(reward.score(torch.randn((1, 3, 1, 4, 4)), ["a"])) == 1.0

    def test_motion_is_a_band_not_a_ramp(self) -> None:
        # An unbounded motion reward is maximised by per-pixel noise: the policy
        # discovers within a few hundred steps that flicker outscores motion.
        reward = MotionMagnitudeReward(target=0.1, tolerance=0.08)
        frame = torch.zeros((1, 1, 1, 4, 4))
        on_target = torch.cat([frame, frame + 0.1, frame + 0.2, frame + 0.3], dim=2)
        wild = torch.randn((1, 1, 4, 4, 4)) * 50.0
        assert float(reward.score(on_target, ["a"])) == pytest.approx(1.0, abs=1e-4)
        assert float(reward.score(wild, ["a"])) < 0.01

    def test_motion_reward_keeps_a_gradient_outside_the_band(self) -> None:
        # A hard window has zero gradient outside it, so a policy that starts
        # outside never learns which direction the band is in.
        reward = MotionMagnitudeReward(target=0.1, tolerance=0.08)
        near = torch.zeros((1, 1, 2, 2, 2))
        far = torch.cat([near, near + 0.4], dim=2)
        nearer = torch.cat([near, near + 0.3], dim=2)
        assert float(reward.score(far, ["a"])) < float(reward.score(nearer, ["a"]))
        assert float(reward.score(far, ["a"])) > 0.0

    def test_av_sync_reports_no_evidence_rather_than_nan(self) -> None:
        # A constant envelope (a static clip, or silence) must correlate to 0
        # rather than to nan, which would poison the whole group's advantage.
        reward = AVSyncProxyReward()
        video = torch.zeros((2, 1, 4, 2, 2))
        audio = torch.zeros((2, 1, 8))
        scores = reward.score({"video": video, "audio": audio}, ["a", "b"])
        assert bool(torch.isfinite(scores).all())
        assert float(scores.abs().max()) == pytest.approx(0.0, abs=1e-5)

    def test_av_sync_without_audio_is_zero(self) -> None:
        reward = AVSyncProxyReward()
        scores = reward.score(torch.randn((2, 1, 4, 2, 2)), ["a", "b"])
        assert torch.equal(scores, torch.zeros(2))

    def test_a_prompt_count_mismatch_raises(self) -> None:
        # A silent misalignment would score every sample against another
        # sample's prompt, which no metric would reveal.
        with pytest.raises(ValueError, match="prompts to match the batch"):
            TemporalConsistencyReward().score(torch.randn((3, 1, 2, 2, 2)), ["a"])

    def test_media_tensors_accepts_tensor_mapping_and_object(self) -> None:
        video = torch.randn((1, 3, 2, 4, 4))
        audio = torch.randn((1, 2, 10))
        assert media_tensors(video)[1] is None
        assert torch.equal(media_tensors({"video": video, "audio": audio})[1], audio)

        class Generated:
            def __init__(self) -> None:
                self.video = video
                self.audio = audio

        assert torch.equal(media_tensors(Generated())[0], video)

    def test_media_tensors_rejects_the_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="must be rank 5"):
            media_tensors(torch.randn((1, 3, 4, 4)))
        with pytest.raises(TypeError, match="media must be"):
            media_tensors("not media")


class TestCompositeReward:
    """Weights must mean relative influence, which needs per-component scaling.

    Without normalisation a reward on the scale of 0.01 and one on the scale of
    40 combined with equal weights is a 1-to-4000 weighting, and nothing in a
    training curve reveals it.
    """

    def test_weights_combine_the_components_exactly(self) -> None:
        media = torch.randn((4, 3, 6, 8, 8))
        prompts = ["a"] * 4
        first, second = TemporalConsistencyReward(), MotionMagnitudeReward()
        composite = CompositeReward(
            components=((first, 2.0), (second, 0.5)), normalization="none"
        )
        expected = (
            first.score(media, prompts) * 2.0 + second.score(media, prompts) * 0.5
        )
        assert torch.equal(composite.score(media, prompts), expected)

    def test_a_zero_weight_removes_a_component(self) -> None:
        media = torch.randn((4, 3, 6, 8, 8))
        prompts = ["a"] * 4
        first, second = TemporalConsistencyReward(), MotionMagnitudeReward()
        composite = CompositeReward(
            components=((first, 1.0), (second, 0.0)), normalization="none"
        )
        assert torch.equal(composite.score(media, prompts), first.score(media, prompts))

    def test_component_scores_are_raw_and_labelled(self) -> None:
        # The diagnostic that matters when a run degrades: the composite can be
        # flat while one component climbs and another collapses.
        media = torch.randn((2, 3, 4, 4, 4))
        prompts = ["a", "b"]
        composite = CompositeReward(
            components=((TemporalConsistencyReward(), 1.0),), normalization="running"
        )
        parts = composite.component_scores(media, prompts)
        assert set(parts) == {"temporal_consistency"}
        torch.testing.assert_close(
            parts["temporal_consistency"],
            TemporalConsistencyReward().score(media, prompts),
        )

    def test_running_normalisation_does_not_use_the_current_batch(self) -> None:
        # The statistics used for a batch must not depend on that batch, or the
        # reward a sample receives depends on which other samples it was scored
        # with and the objective stops being a function of the sample.
        media = torch.randn((4, 3, 4, 4, 4))
        prompts = ["a"] * 4
        raw = TemporalConsistencyReward()
        composite = CompositeReward(components=((raw, 1.0),), normalization="running")
        assert torch.equal(composite.score(media, prompts), raw.score(media, prompts))
        # The second batch is normalised against the first one's statistics.
        assert not torch.equal(
            composite.score(media, prompts), raw.score(media, prompts)
        )

    def test_batch_normalisation_standardises_within_the_batch(self) -> None:
        media = torch.randn((8, 3, 4, 4, 4))
        prompts = ["a"] * 8
        composite = CompositeReward(
            components=((MotionMagnitudeReward(), 1.0),), normalization="batch"
        )
        scores = composite.score(media, prompts)
        assert float(scores.mean().abs()) < 1e-4

    def test_state_round_trips_through_a_checkpoint(self) -> None:
        # A normaliser that resets on resume changes the effective weighting of
        # every component at the restart, and the discontinuity in the loss curve
        # is indistinguishable from a real training problem.
        media = torch.randn((4, 3, 4, 4, 4))
        prompts = ["a"] * 4
        composite = CompositeReward(components=((TemporalConsistencyReward(), 1.0),))
        composite.score(media, prompts)
        state = composite.state_dict()
        restored = CompositeReward(components=((TemporalConsistencyReward(), 1.0),))
        restored.load_state_dict(state)
        assert torch.equal(
            restored.score(media, prompts), composite.score(media, prompts)
        )

    def test_configuration_errors_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one component"):
            CompositeReward(components=())
        with pytest.raises(ValueError, match="normalization must be"):
            CompositeReward(
                components=((TemporalConsistencyReward(), 1.0),), normalization="zscore"
            )
        with pytest.raises(KeyError, match="missing normalizer state"):
            CompositeReward(
                components=((TemporalConsistencyReward(), 1.0),)
            ).load_state_dict({})


class TestRunningNormalizer:
    """Streaming statistics must be exact for batched updates.

    Welford/Chan rather than accumulated sums of squares, because a reward stream
    is long and the naive form loses precision exactly when the variance is
    small — which is the regime where the normaliser matters most.
    """

    def test_the_first_batch_is_never_divided_by_nothing(self) -> None:
        normalizer = RunningNormalizer()
        values = torch.tensor([1.0, 2.0])
        assert torch.equal(normalizer.normalize(values), values)

    def test_batched_updates_match_the_full_population_statistics(self) -> None:
        normalizer = RunningNormalizer()
        values = torch.randn(256)
        for chunk in values.split(37):
            normalizer.update(chunk)
        state = normalizer.state_dict()
        assert state["mean"] == pytest.approx(float(values.mean()), abs=1e-5)
        assert state["variance"] == pytest.approx(
            float(values.var(unbiased=False)), abs=1e-4
        )

    def test_normalising_centres_and_scales(self) -> None:
        normalizer = RunningNormalizer()
        normalizer.update(torch.tensor([0.0, 2.0, 4.0, 6.0]))
        assert float(normalizer.normalize(torch.tensor([3.0]))) == pytest.approx(
            0.0, abs=1e-5
        )

    def test_momentum_tracks_a_drifting_distribution(self) -> None:
        normalizer = RunningNormalizer(momentum=0.9)
        normalizer.update(torch.zeros(8))
        for _ in range(50):
            normalizer.update(torch.full((8,), 10.0))
        assert normalizer.state_dict()["mean"] > 9.0

    def test_an_empty_update_is_a_no_op(self) -> None:
        normalizer = RunningNormalizer()
        normalizer.update(torch.empty(0))
        assert normalizer.state_dict()["count"] == 0.0

    def test_state_round_trips(self) -> None:
        normalizer = RunningNormalizer()
        normalizer.update(torch.randn(32))
        restored = RunningNormalizer()
        restored.load_state_dict(normalizer.state_dict())
        assert restored.state_dict() == normalizer.state_dict()

    def test_invalid_construction_and_state_raise(self) -> None:
        with pytest.raises(ValueError, match="momentum must be"):
            RunningNormalizer(momentum=1.0)
        with pytest.raises(ValueError, match="epsilon must be"):
            RunningNormalizer(epsilon=0.0)
        with pytest.raises(KeyError, match="missing normalizer statistic"):
            RunningNormalizer().load_state_dict({"mean": 0.0})


class TestDPOLoss:
    """An identical pair carries no preference, so it must produce no signal.

    With shared noise, an identical winner and loser give identical errors on
    both the policy and the reference, the margin is exactly zero, and the
    centred loss is exactly zero. That exactness is what makes ``log 2``
    centring worth doing: zero on the logged curve means "at chance" rather than
    0.693.
    """

    def test_an_identical_pair_gives_exactly_zero(self) -> None:
        error = torch.tensor([0.3, 0.5])
        loss, margin = dpo_loss(error, error, error, error, DPOConfig())
        assert float(loss) == 0.0
        assert torch.equal(margin.abs(), torch.zeros(2))

    def test_without_centring_an_identical_pair_costs_log_two(self) -> None:
        error = torch.tensor([0.3, 0.5])
        loss, _ = dpo_loss(error, error, error, error, DPOConfig(center_loss=False))
        assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)

    def test_a_policy_that_prefers_the_winner_has_a_positive_margin(self) -> None:
        # The policy's error is lower on the winner than the reference's is, so
        # the logit is positive and the loss is below chance.
        reference = torch.tensor([0.5, 0.5])
        loss, margin = dpo_loss(
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.8, 0.8]),
            reference,
            reference,
            DPOConfig(beta=1.0),
        )
        assert bool((margin > 0.0).all())
        assert float(loss) < 0.0

    def test_label_smoothing_bounds_a_confidently_wrong_pair(self) -> None:
        # Human preference data has a real, irreducible label-noise rate, and an
        # unsmoothed loss grows without bound on a mislabelled pair.
        winner = torch.tensor([5.0])
        loser = torch.tensor([0.0])
        reference = torch.tensor([0.0])
        sharp, _ = dpo_loss(
            winner, loser, reference, reference, DPOConfig(beta=1.0, center_loss=False)
        )
        smoothed, _ = dpo_loss(
            winner,
            loser,
            reference,
            reference,
            DPOConfig(beta=1.0, center_loss=False, label_smoothing=0.2),
        )
        assert float(smoothed) < float(sharp)

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo"])
    def test_every_variant_is_finite(self, loss_type: str) -> None:
        loss, margin = dpo_loss(
            torch.tensor([0.2, 0.4]),
            torch.tensor([0.5, 0.3]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.3, 0.3]),
            DPOConfig(beta=1.0, loss_type=loss_type),
        )
        assert bool(torch.isfinite(loss))
        assert bool(torch.isfinite(margin).all())

    def test_hinge_stops_rewarding_a_separated_pair(self) -> None:
        # No reward once the pair is separated by one unit, which stops a
        # well-separated pair from dominating the batch gradient forever.
        reference = torch.tensor([0.0])
        config = DPOConfig(beta=1.0, loss_type="hinge")
        separated, _ = dpo_loss(
            torch.tensor([-5.0]), torch.tensor([0.0]), reference, reference, config
        )
        further, _ = dpo_loss(
            torch.tensor([-50.0]), torch.tensor([0.0]), reference, reference, config
        )
        assert float(separated) == 0.0
        assert float(further) == 0.0

    def test_beta_scales_with_the_step_count_only_when_asked(self) -> None:
        # The published beta values already absorb the factor, so applying it by
        # default would saturate the sigmoid on the first batch.
        assert DPOConfig(beta=2.0).effective_beta == pytest.approx(2.0)
        assert DPOConfig(
            beta=2.0, beta_scales_with_steps=True, timestep_count=10
        ).effective_beta == pytest.approx(20.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"beta": 0.0}, "beta must be"),
            ({"loss_type": "kto"}, "loss_type must be"),
            ({"label_smoothing": 0.5}, "label_smoothing must be"),
            ({"timestep_count": 0}, "timestep_count must be"),
        ],
    )
    def test_invalid_fields_raise(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            DPOConfig(**kwargs)


class TestDPOObjective:
    """DPO must drop into the ordinary trainer, which is the point of the class.

    ``DPOObjective`` satisfies :class:`avgen.train.Objective`, so preference
    tuning needs no RL trainer, no rollout buffer and no new loop: it is
    ``Trainer(state, DPOObjective(...), ...)`` and checkpointing, EMA, gradient
    accumulation, context parallelism and telemetry all apply unchanged. A
    bespoke trainer would be a second code path to keep in sync with the first,
    and the second one is always the one that rots.
    """

    def _pair_batch(self, model: VideoDiT, *, identical: bool) -> MediaBatch:
        torch.manual_seed(9)
        clean = torch.randn((2, model.config.in_channels, FRAMES, EXTENT, EXTENT))
        loser = clean if identical else clean + 0.3 * torch.randn_like(clean)
        return make_media_batch(torch.cat((clean, loser)), model.config.text_width)

    def test_it_satisfies_the_objective_protocol(self) -> None:
        model = build_model()
        objective = DPOObjective(DPOConfig(), copy.deepcopy(model))
        assert isinstance(objective, Objective)
        # isinstance on a runtime-checkable Protocol only checks that the
        # attribute exists, so the signature is compared explicitly: this is
        # what actually lets the trainer call it.
        expected = list(inspect.signature(Objective.__call__).parameters)[1:]
        actual = list(inspect.signature(objective.__call__).parameters)
        assert actual == expected

    def test_it_returns_the_framework_objective_output(self) -> None:
        model = build_model()
        objective = DPOObjective(DPOConfig(), copy.deepcopy(model))
        output = objective(
            model,
            self._pair_batch(model, identical=True),
            RNGStreams.from_seed(3),
            patchifier=model.patchifier,
        )
        assert isinstance(output, ObjectiveOutput)
        assert output.loss.dtype is torch.float32
        # DPO is a video-only objective; the audio component is a real zero
        # rather than an omitted field, so the trainer's reduction is unchanged.
        assert float(output.audio_loss) == 0.0

    def test_an_identical_pair_produces_exactly_no_signal(self) -> None:
        model = build_model()
        objective = DPOObjective(DPOConfig(), copy.deepcopy(model))
        output = objective(
            model,
            self._pair_batch(model, identical=True),
            RNGStreams.from_seed(3),
            patchifier=model.patchifier,
        )
        assert float(output.loss.detach()) == 0.0
        assert objective.last_margin is not None
        assert torch.equal(objective.last_margin.abs(), torch.zeros(2))

    def test_a_policy_equal_to_its_reference_starts_at_chance(self) -> None:
        # At initialisation the policy *is* the reference, so the two error
        # differences cancel exactly and the margin is zero — for a genuinely
        # different pair too. This pins the starting point of every DPO run.
        model = build_model()
        objective = DPOObjective(DPOConfig(), copy.deepcopy(model))
        output = objective(
            model,
            self._pair_batch(model, identical=False),
            RNGStreams.from_seed(3),
            patchifier=model.patchifier,
        )
        assert float(output.loss.detach()) == 0.0

    def test_a_diverged_policy_gives_a_finite_loss_that_backpropagates(self) -> None:
        model = build_model()
        reference = copy.deepcopy(model)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(21)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)

        objective = DPOObjective(DPOConfig(beta=100.0), reference)
        output = objective(
            model,
            self._pair_batch(model, identical=False),
            RNGStreams.from_seed(3),
            patchifier=model.patchifier,
        )
        assert bool(torch.isfinite(output.loss))
        assert float(output.loss.detach()) > 0.0
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients
        assert all(bool(torch.isfinite(g).all()) for g in gradients)
        # A finite loss with an all-zero gradient would train nothing.
        assert sum(float(g.abs().sum()) for g in gradients) > 0.0
        # The reference is frozen and must stay that way.
        assert all(p.grad is None for p in reference.parameters())

    @pytest.mark.parametrize("loss_type", ["hinge", "ipo"])
    def test_the_variants_run_end_to_end(self, loss_type: str) -> None:
        model = build_model()
        objective = DPOObjective(
            DPOConfig(beta=1.0, loss_type=loss_type), copy.deepcopy(model)
        )
        output = objective(
            model,
            self._pair_batch(model, identical=False),
            RNGStreams.from_seed(1),
            patchifier=model.patchifier,
        )
        assert bool(torch.isfinite(output.loss))

    def test_snr_weighting_runs_and_changes_the_loss(self) -> None:
        model = build_model()
        reference = copy.deepcopy(model)
        torch.manual_seed(21)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        batch = self._pair_batch(model, identical=False)
        plain = DPOObjective(DPOConfig(beta=100.0), reference)(
            model, batch, RNGStreams.from_seed(1), patchifier=model.patchifier
        )
        weighted = DPOObjective(DPOConfig(beta=100.0, snr_weighting=True), reference)(
            model, batch, RNGStreams.from_seed(1), patchifier=model.patchifier
        )
        assert bool(torch.isfinite(weighted.loss))
        assert float(plain.loss.detach()) != float(weighted.loss.detach())

    def test_an_odd_batch_raises(self) -> None:
        # Pairs that do not line up would compute every preference against the
        # wrong partner, which no metric would reveal.
        model = build_model()
        objective = DPOObjective(DPOConfig(), copy.deepcopy(model))
        torch.manual_seed(9)
        video = torch.randn((3, model.config.in_channels, FRAMES, EXTENT, EXTENT))
        batch = make_media_batch(torch.cat((video, video[:1])), model.config.text_width)
        odd = MediaBatch(
            video=batch.video[:3],
            audio=batch.audio[:3],
            text=batch.text[:3],
            video_mask=batch.video_mask[:3],
            audio_mask=batch.audio_mask[:3],
            video_positions=batch.video_positions[:3],
            audio_positions=batch.audio_positions[:3],
            sample_ids=batch.sample_ids[:3],
            spec=batch.spec,
            text_mask=batch.text_mask[:3],
        )
        with pytest.raises(ValueError, match="even batch of stacked"):
            objective(model, odd, RNGStreams.from_seed(0), patchifier=model.patchifier)

    def test_a_custom_timestep_sampler_is_used(self) -> None:
        class FixedSampler:
            def __init__(self) -> None:
                self.calls = 0

            def sample(
                self, batch: int, *, device: Any, generator: Any = None, **_: Any
            ) -> torch.Tensor:
                del generator
                self.calls += 1
                return torch.full((batch,), 0.5, device=device)

        model = build_model()
        sampler = FixedSampler()
        objective = DPOObjective(
            DPOConfig(), copy.deepcopy(model), timestep_sampler=sampler
        )
        objective(
            model,
            self._pair_batch(model, identical=True),
            RNGStreams.from_seed(0),
            patchifier=model.patchifier,
        )
        assert sampler.calls == 1


class TestGRPOConfig:
    """A deterministic policy has no gradient, so zero noise must be rejected.

    The clip range defaults are also worth pinning: 2e-6 is the value GRPO-Guard
    reports for the *normalised* ratio. PPO's familiar 0.2 is not a meaningful
    value for a diffusion policy at either scale, and silently accepting it would
    make the clip never bind.
    """

    def test_the_upper_clip_mirrors_the_lower_one_by_default(self) -> None:
        assert GRPOConfig(clip_range=1e-4).high == pytest.approx(1e-4)
        # DAPO's clip-higher: an asymmetric range gives low-probability
        # improvements more room, counteracting entropy collapse.
        assert GRPOConfig(clip_range=1e-4, clip_range_high=3e-4).high == pytest.approx(
            3e-4
        )

    def test_a_deterministic_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero entropy and zero policy gradient"):
            GRPOConfig(noise_level=0.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"group_size": 1}, "at least 2"),
            ({"clip_range": 0.0}, "clip_range must be"),
            ({"clip_range_high": 0.0}, "clip_range_high must be"),
            ({"kl_coefficient": -1.0}, "kl_coefficient must be"),
            ({"inner_epochs": 0}, "inner_epochs must be"),
            ({"minibatch_size": 0}, "minibatch_size must be"),
            ({"max_grad_norm": 0.0}, "max_grad_norm must be"),
        ],
    )
    def test_invalid_fields_raise(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            GRPOConfig(**kwargs)

    def test_a_kl_penalty_without_a_reference_is_refused(self) -> None:
        # A KL penalty needs something to be relative to; defaulting to the
        # current policy would make the term identically zero and silently
        # disable a regulariser someone asked for.
        model = build_model()
        state = TrainState(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            rng=RNGStreams.from_seed(0),
        )
        with pytest.raises(ValueError, match="no reference policy"):
            GRPOTrainer(
                state,
                make_velocity_fn(model, torch.randn((1, TEXT_TOKENS, 32))),
                GRPOConfig(kl_coefficient=0.04),
            )


class TestGRPOEndToEnd:
    """A whole GRPO round against the real model, on CPU, in a few seconds.

    This is the test that would catch a break in the seam between the sampler,
    the buffer and the update — which is where the subtle failures live, because
    each component is individually correct and the composition is what carries
    the on-policy assumption.
    """

    def _trainer(
        self,
        *,
        group_size: int = 3,
        minibatch_size: int = 3,
        prompts: int = 2,
        **config_kwargs: Any,
    ) -> tuple[GRPOTrainer, VideoDiT, TrainState]:
        model = build_model()
        torch.manual_seed(13)
        features = torch.randn((prompts, TEXT_TOKENS, model.config.text_width))
        state = TrainState(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            rng=RNGStreams.from_seed(0),
        )
        config = GRPOConfig(
            group_size=group_size,
            minibatch_size=minibatch_size,
            clip_range=1e-3,
            noise_level=0.7,
            window=WindowSchedule(window=2, stride=1, interval=2, total_steps=4),
            **config_kwargs,
        )
        trainer = GRPOTrainer(state, make_velocity_fn(model, features), config)
        return trainer, model, state

    def _reward(self) -> CompositeReward:
        return CompositeReward(
            components=(
                (TemporalConsistencyReward(), 1.0),
                (MotionMagnitudeReward(), 0.5),
            ),
            normalization="none",
        )

    def test_a_rollout_stores_only_the_window(self) -> None:
        trainer, model, _ = self._trainer()
        assert trainer.gradient_steps() == (0, 1)
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        # Four denoising steps, two inside the window; six trajectories.
        assert len(buffer) == 2
        assert buffer.num_samples == 6
        assert buffer.rewards is not None
        assert bool(torch.isfinite(buffer.rewards).all())
        assert buffer.prompts == ("a cat", "a dog")

    def test_the_first_pass_ratio_is_exactly_one(self) -> None:
        # Nothing has changed yet. The regulated log ratio is -(dmu . eps) and
        # dmu is exactly zero, so the ratio is exactly 1.0 and nothing clips.
        # Any drift means the update is scoring a transition the rollout did not
        # take.
        trainer, model, _ = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(6, shuffle=False))
        loss, metrics = trainer.loss(batch)
        assert float(metrics.ratio_mean) == 1.0
        assert float(metrics.clip_fraction) == 0.0
        assert bool(torch.isfinite(loss))

    def test_advantages_are_zero_mean_within_each_prompt_group(self) -> None:
        trainer, model, _ = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        advantages = buffer.compute_advantages()
        means = advantages.reshape(2, 3).mean(dim=1)
        assert bool(means.abs().max() < 1e-5)

    def test_an_update_is_finite_and_moves_the_parameters(self) -> None:
        trainer, model, state = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        before = [p.detach().clone() for p in model.parameters()]
        collected = trainer.update(buffer)

        assert collected
        assert all(bool(torch.isfinite(m.loss)) for m in collected)
        assert all(bool(torch.isfinite(m.policy_loss)) for m in collected)
        assert state.step == len(collected)
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before, model.parameters(), strict=True)
        )
        # Gradients are released after each step; a retained graph here would be
        # the memory leak that ends a real run.
        assert all(p.grad is None for p in model.parameters())

    def test_metrics_reduce_to_host_floats(self) -> None:
        trainer, model, _ = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        mapping = trainer.update(buffer)[0].to_mapping()
        assert set(mapping) == {
            "grpo/loss",
            "grpo/policy_loss",
            "grpo/kl",
            "grpo/clip_fraction",
            "grpo/ratio_mean",
            "grpo/advantage_mean",
            "grpo/advantage_std",
        }
        assert all(isinstance(value, float) for value in mapping.values())
        assert all(math.isfinite(value) for value in mapping.values())
        # A minibatch is a shuffled subset of the (trajectory, step) grid rather
        # than a whole group, so its advantage mean is not zero — the
        # zero-mean-per-group property is asserted on the full buffer above.
        assert mapping["grpo/clip_fraction"] >= 0.0

    def test_inner_epochs_multiply_the_optimizer_steps(self) -> None:
        # More than one pass makes the update off-policy, which is exactly what
        # the ratio and the clipping exist to permit.
        trainer, model, state = self._trainer(minibatch_size=6, inner_epochs=3)
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        assert len(trainer.update(buffer)) == 3 * 2  # 3 epochs x 2 stored steps
        assert state.step == 6

    def test_updating_without_rewards_raises(self) -> None:
        trainer, _, _ = self._trainer()
        with pytest.raises(RuntimeError, match="set_rewards"):
            trainer.update(RolloutBuffer(group_size=3))

    def test_the_kl_term_is_positive_against_a_divergent_reference(self) -> None:
        model = build_model()
        torch.manual_seed(13)
        features = torch.randn((2, TEXT_TOKENS, model.config.text_width))
        reference_model = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in reference_model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)
        state = TrainState(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            rng=RNGStreams.from_seed(0),
        )
        base = GRPOConfig(
            group_size=3, minibatch_size=6, clip_range=1e-3, noise_level=0.7
        )
        rollout_trainer = GRPOTrainer(state, make_velocity_fn(model, features), base)
        buffer = rollout_trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 4),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(6, shuffle=False))

        penalised = GRPOTrainer(
            state,
            make_velocity_fn(model, features),
            GRPOConfig(
                group_size=3,
                minibatch_size=6,
                clip_range=1e-3,
                noise_level=0.7,
                kl_coefficient=0.04,
                regulated_clip=False,
            ),
            reference_velocity_fn=make_velocity_fn(reference_model, features),
        )
        loss, metrics = penalised.loss(batch)
        assert bool(torch.isfinite(loss))
        assert float(metrics.kl) > 0.0
        # The penalty is added to the loss, so it must raise it.
        assert float(loss.detach()) > float(metrics.policy_loss)

    def test_the_kl_term_is_absent_when_disabled(self) -> None:
        trainer, model, _ = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(6, shuffle=False))
        _, metrics = trainer.loss(batch)
        assert float(metrics.kl) == 0.0

    def test_the_raw_ratio_path_also_starts_near_one(self) -> None:
        # With regulated_clip off the ratio is exp(log pi_theta - log pi_old),
        # which is exactly 1 on the first pass because the replay is exact.
        trainer, model, _ = self._trainer(regulated_clip=False)
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(6, shuffle=False))
        _, metrics = trainer.loss(batch)
        assert float(metrics.ratio_mean) == pytest.approx(1.0, abs=1e-5)

    def test_timestep_weights_average_to_one(self) -> None:
        # GRPO-Guard's 1/dt, renormalised so the loss magnitude does not depend
        # on the rollout step count — otherwise changing the discretisation
        # silently changes the effective learning rate.
        trainer, model, _ = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.tensor([1.0, 0.9, 0.5, 0.2, 0.0]),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(12, shuffle=False))
        weights = trainer._step_weights(batch)
        assert weights is not None
        assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)

    def test_rollout_velocity_batch_evaluates_with_gradient(self) -> None:
        trainer, model, _ = self._trainer()
        buffer = trainer.rollout(
            ["a cat", "a dog"],
            sigmas=torch.linspace(1.0, 0.0, 5),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            reward=self._reward(),
        )
        buffer.compute_advantages()
        batch = next(buffer.iter_minibatches(6, shuffle=False))
        velocity = rollout_velocity_batch(trainer.velocity_fn, batch)
        assert velocity.shape == batch.sample_before.shape
        assert velocity.requires_grad


class TestModelVelocityFn:
    """The seam between the RL package and a real avgen model.

    ``model_velocity_fn`` closes over one batch's conditioning, so it is the
    rollout-side adapter: it assembles a ``ModelInput`` per call and unfolds the
    token-space output back into the latent grid the sampler works in.
    """

    def test_it_produces_a_velocity_shaped_like_the_latents(self) -> None:
        model = build_model()
        batch = 4
        positions = torch.arange(FRAMES, dtype=torch.float32).repeat(batch, 1) / 8.0
        mask = torch.ones((batch, FRAMES, EXTENT, EXTENT), dtype=torch.bool)
        velocity_fn = model_velocity_fn(
            model,
            patchifier=model.patchifier,
            positions=positions,
            mask=mask,
            text_features=torch.randn((batch, TEXT_TOKENS, model.config.text_width)),
            text_mask=torch.ones((batch, TEXT_TOKENS), dtype=torch.bool),
        )
        latents = torch.randn((batch, model.config.in_channels, FRAMES, EXTENT, EXTENT))
        velocity = velocity_fn(
            latents, torch.rand(batch), prompt_index=torch.arange(batch)
        )
        assert velocity.shape == latents.shape
        assert bool(torch.isfinite(velocity).all())

    def test_it_drives_a_rollout_against_the_real_model(self) -> None:
        model = build_model()
        batch = 4  # 2 prompts x group of 2
        positions = torch.arange(FRAMES, dtype=torch.float32).repeat(batch, 1) / 8.0
        mask = torch.ones((batch, FRAMES, EXTENT, EXTENT), dtype=torch.bool)
        velocity_fn = model_velocity_fn(
            model,
            patchifier=model.patchifier,
            positions=positions,
            mask=mask,
            text_features=torch.randn((batch, TEXT_TOKENS, model.config.text_width)),
            text_mask=torch.ones((batch, TEXT_TOKENS), dtype=torch.bool),
        )
        buffer = generate_group(
            velocity_fn,
            prompts=["a cat", "a dog"],
            group_size=2,
            sigmas=torch.linspace(1.0, 0.0, 4),
            latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
            generator=torch.Generator().manual_seed(0),
        )
        assert buffer.media is not None
        assert bool(torch.isfinite(buffer.media).all())


@pytest.mark.gpu
def test_a_rollout_runs_on_cuda() -> None:
    """The rollout path must not have a hidden host-side dependency."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = build_model().cuda()
    patchifier = model.patchifier

    def velocity(
        sample: torch.Tensor, sigma: torch.Tensor, *, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        del prompt_index
        count = sample.shape[0]
        positions = (
            torch.arange(FRAMES, dtype=torch.float32, device=sample.device).repeat(
                count, 1
            )
            / 8.0
        )
        mask = torch.ones(
            (count, FRAMES, EXTENT, EXTENT), dtype=torch.bool, device=sample.device
        )
        stream = patchifier.to_tokens(
            sample, positions=positions, mask=mask, noise_level=sigma
        )
        inputs = ModelInput(
            video=stream,
            audio=TokenStream.empty_like(count, stream.width, device=sample.device),
            text=TextContext(
                features=torch.randn(
                    (count, TEXT_TOKENS, model.config.text_width),
                    device=sample.device,
                ),
                mask=torch.ones(
                    (count, TEXT_TOKENS), dtype=torch.bool, device=sample.device
                ),
            ),
        )
        return unpatchify_grid(model(inputs).video, stream.layout)

    buffer = generate_group(
        velocity,
        prompts=["a cat"],
        group_size=2,
        sigmas=torch.linspace(1.0, 0.0, 4),
        latent_shape=(model.config.in_channels, FRAMES, EXTENT, EXTENT),
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    assert buffer.media is not None
    assert buffer.media.device.type == "cuda"


def test_the_patchifier_used_here_is_the_model_s_own() -> None:
    """The RL tests drive the model through the geometry it was built for."""
    model = build_model()
    assert isinstance(model.patchifier, GridPatchifier)
