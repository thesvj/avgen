"""Tests for inference: schedules, samplers, guidance, conditioning, the pipeline.

The inference subsystem is where a correct model produces wrong samples, and the
ways that happens are specific enough to test directly rather than by eyeballing
output. Four properties carry most of this file.

**Solver correctness.** For rectified flow the ODE is exactly solvable, so every
sampler has a known answer to integrate towards and "close enough" is not the
standard. Feeding a solver a genuine velocity field whose exact trajectory is
known in closed form turns a whole class of sign errors, alpha/sigma mixups, and
misplaced ``1 -`` into a numeric failure instead of a slightly-worse sample
nobody attributes to the sampler.

**Schedule/training agreement.** The inference shift must equal the training
shift. Sampling a ``shift=3`` checkpoint on an unshifted schedule steps through
noise levels at a density the model was never trained for, and the result is
systematically over- or under-denoised — washed out when the shift is too low,
oversaturated when it is too high. It is not subtle and it is not visible in the
sampler, so the two implementations are asserted equal numerically.

**Conditioning identity.** avgen claims no train/inference skew because the
inference conditioning builders perform the *same* three steps in the same order
through the same patchifier as the training objective. That claim is worth
exactly as much as a test that runs both and compares tensors, so
``TestConditioningMatchesTraining`` does precisely that — bit-identical, not
approximately.

**Guidance identities.** Scale 1.0 must be bit-identical to unguided, or a user
cannot turn guidance off without changing the sample; and the repairs
(CFG-rescale, APG) must actually do what they claim to the magnitude.

Everything runs on CPU at tiny shapes in a few seconds. The ``gpu``-marked tests
only re-check device placement and device-RNG determinism.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from avgen.codecs import (
    ReferenceAudioCodec,
    ReferenceTextEncoder,
    ReferenceVideoCodec,
)
from avgen.core import (
    ConditionMode,
    GridPatchifier,
    MediaBatch,
    MediaBatchSpec,
    RNGStreams,
)
from avgen.core.model_input import ModelInput, ModelOutput
from avgen.core.tokens import TextContext
from avgen.infer import (
    EulerAncestralSampler,
    GenerationConfig,
    GenerationPipeline,
    GuidanceConfig,
    SamplerConfig,
    ScheduleConfig,
    SigmaSchedule,
    StreamState,
    adaptive_projected_guidance,
    apply_guidance,
    apply_modality_guidance,
    apply_shift,
    build_model_input,
    build_sampler,
    build_sigma_schedule,
    classifier_free_guidance,
    condition_first_frame,
    condition_mask,
    condition_stream,
    condition_temporal_prefix,
    default_audio_patchifier,
    first_frame_mask,
    karras_sigmas,
    linear_quadratic_sigmas,
    linear_sigmas,
    list_samplers,
    list_sigma_schedules,
    register_sampler,
    register_sigma_schedule,
    rescale_guidance,
    resolution_shift,
    to_epsilon,
    to_x0,
    unconditional_input,
)
from avgen.models import VideoDiT, preset
from avgen.train import FlowMatchingConfig, FlowMatchingObjective
from avgen.train.conditioning import ConditioningPlan
from avgen.train.timestep import ShiftedLogitNormalSampler, shift_timesteps
from avgen.train.timestep import resolution_shift as train_resolution_shift

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: Tolerance for a solver that should be exact on the field it is given. The
#: observed worst case across every registered sampler and schedule is ~1e-6 in
#: float32; anything an order of magnitude above that is a formula error, not
#: accumulated round-off.
SOLVER_EXACT = 1e-5


def constant_field(velocity: torch.Tensor):
    """Return a denoiser for the flow between one fixed pair ``(x0, eps)``.

    The rectified-flow path between a fixed clean sample and a fixed noise draw
    is a straight line, so its velocity ``eps - x0`` is constant in both ``x``
    and ``t``. The exact solution from any point on the line is therefore
    available in closed form, which is what makes the solver assertions below
    statements about the solver rather than about a model.

    Args:
        velocity: The constant field value.

    Returns:
        A ``(x, sigma) -> velocity`` callable.
    """

    def denoise(probe: torch.Tensor, at: float) -> torch.Tensor:
        del probe, at
        return velocity

    return denoise


def dirac_field(clean: torch.Tensor):
    """Return the exact marginal velocity field for a single data point.

    With the data distribution a point mass at ``clean``, conditioning on
    ``x_t`` determines the noise exactly, and the marginal velocity collapses to
    ``(x - clean) / t``. Unlike :func:`constant_field` this genuinely depends on
    ``x``, so a solver that ignored its input or evaluated at the wrong point
    would still pass the constant-field test and fails this one.

    Every trajectory is a straight line through ``clean``, so the exact solution
    is ``x(t) = clean + (t / t0) * (x(t0) - clean)``.

    Args:
        clean: The data point.

    Returns:
        A ``(x, sigma) -> velocity`` callable.
    """

    def denoise(probe: torch.Tensor, at: float) -> torch.Tensor:
        # The field has a pole at t = 0, where no schedule ever evaluates; the
        # floor keeps the callback total without changing any result.
        return (probe - clean) / max(at, 1e-7)

    return denoise


def integrate(
    name: str,
    denoise: Any,
    start: torch.Tensor,
    schedule: SigmaSchedule,
    *,
    eta: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Run a named sampler over a whole schedule against a known field."""
    solver = build_sampler(SamplerConfig(name=name, eta=eta))
    solver.reset()
    generator = torch.Generator().manual_seed(seed)
    current = start
    for _, sigma, sigma_next in schedule:
        current = solver.step(
            denoise(current, sigma),
            current,
            sigma,
            sigma_next,
            denoise=denoise,
            generator=generator,
        )
    return current


class TestSigmaSchedules:
    """A schedule is a strictly decreasing path from noise to exactly zero.

    Stopping short leaves residual noise that no later stage removes: the
    difference between a final sigma of 0.0 and one of 0.02 is a faint grain
    over the whole frame. Monotonicity matters for the opposite reason — a flat
    or rising step makes the solver stall or run backwards, and several of the
    solvers divide by the step size.
    """

    def test_every_registered_family_is_buildable(self) -> None:
        assert set(list_sigma_schedules()) == {"karras", "linear", "linear_quadratic"}

    @pytest.mark.parametrize("name", list_sigma_schedules())
    @pytest.mark.parametrize("steps", [2, 3, 8, 30])
    def test_every_family_honours_the_schedule_contract(
        self, name: str, steps: int
    ) -> None:
        schedule = build_sigma_schedule(ScheduleConfig(name=name, steps=steps))
        sigmas = schedule.sigmas
        assert schedule.num_steps == steps
        assert int(sigmas.numel()) == steps + 1
        assert float(sigmas[-1]) == 0.0, "a non-zero tail leaves visible grain"
        assert float(sigmas[0]) <= 1.0, "a flow sigma above 1 is an EDM sigma"
        assert bool((sigmas[1:] - sigmas[:-1] < 0).all())

    def test_linear_is_uniform_in_flow_time(self) -> None:
        # The probability path is a straight line, so uniform spacing in t is
        # uniform spacing along the path — including the final jump to zero,
        # which the default sigma_min is chosen to make the same size as the
        # rest.
        sigmas = linear_sigmas(4, sigma_max=1.0)
        torch.testing.assert_close(
            sigmas, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), rtol=0, atol=1e-6
        )

    def test_karras_concentrates_at_low_noise(self) -> None:
        # Its bias is the opposite of the shift's, which is worth knowing before
        # tuning both at once.
        sigmas = karras_sigmas(16, sigma_max=1.0)
        gaps = (sigmas[:-1] - sigmas[1:])[:-1]
        assert float(gaps[0]) > float(gaps[-1])

    def test_linear_quadratic_takes_smaller_bites_near_the_clean_end(self) -> None:
        # Designed for the few-step regime, where the solver's error is
        # dominated by the low-noise end as the sample commits to detail.
        sigmas = linear_quadratic_sigmas(8, sigma_max=1.0, threshold_noise=0.025)
        gaps = sigmas[:-1] - sigmas[1:]
        assert float(gaps[0]) > float(gaps[-1])

    def test_the_shift_reparametrises_without_moving_the_endpoints(self) -> None:
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        shifted = apply_shift(sigmas, 3.0)
        assert float(shifted[0]) == pytest.approx(1.0)
        assert float(shifted[-1]) == pytest.approx(0.0)
        # shift > 1 pushes interior values up, concentrating the schedule where
        # the video's global structure is decided.
        assert float(shifted[1]) > 0.5

    def test_shift_of_one_is_the_identity_object(self) -> None:
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        assert apply_shift(sigmas, 1.0) is sigmas

    def test_a_shifted_schedule_still_ends_at_exactly_zero(self) -> None:
        # apply_shift fixes 0 exactly in real arithmetic; in float32 it can drift
        # by an ulp, and the schedule contract admits no drift.
        schedule = build_sigma_schedule(
            ScheduleConfig(name="linear", steps=7, shift=5.0)
        )
        assert float(schedule.sigmas[-1]) == 0.0
        assert schedule.shift == 5.0

    def test_iteration_yields_python_floats_not_device_tensors(self) -> None:
        # A sampler that indexed a device tensor per step would force a host
        # synchronisation on every step to make the value usable in control flow.
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=3))
        steps = list(schedule)
        assert [index for index, _, _ in steps] == [0, 1, 2]
        assert all(isinstance(sigma, float) for _, sigma, _ in steps)
        assert steps[-1][2] == 0.0

    def test_timesteps_excludes_the_terminal_zero(self) -> None:
        # No model evaluation happens at sigma = 0, so counting it would report
        # one more forward pass than the run actually makes.
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=5))
        assert int(schedule.timesteps().numel()) == 5

    @pytest.mark.parametrize(
        "sigmas",
        [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0.5]),
            torch.tensor([1.0, 0.5, 0.5, 0.0]),
            torch.tensor([1.0, 0.5, 0.02]),
            torch.tensor([1.5, 0.5, 0.0]),
            torch.tensor([1.0, math.nan, 0.0]),
        ],
    )
    def test_rejects_a_schedule_no_solver_could_traverse(
        self, sigmas: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError):
            SigmaSchedule(sigmas=sigmas)

    def test_rejects_an_integer_schedule(self) -> None:
        with pytest.raises(TypeError, match="floating point"):
            SigmaSchedule(sigmas=torch.tensor([1, 0]))

    def test_an_unknown_family_names_what_is_registered(self) -> None:
        with pytest.raises(KeyError, match="unknown sigma schedule"):
            build_sigma_schedule(ScheduleConfig(name="ddim"))

    def test_re_registering_a_name_is_refused(self) -> None:
        # A config naming a schedule must resolve to the same curve for the life
        # of a checkpoint, so silent replacement is not allowed.
        with pytest.raises(ValueError, match="already registered"):
            register_sigma_schedule("linear")

    def test_survives_a_serialisation_round_trip(self) -> None:
        config = ScheduleConfig(name="karras", steps=12, shift=2.5, rho=5.0)
        assert ScheduleConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"steps": 0},
            {"name": ""},
            {"shift": 0.0},
            {"shift": math.inf},
        ],
    )
    def test_rejects_an_invalid_config(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ScheduleConfig(**kwargs)

    # A one-step schedule is exactly what a distilled few-step checkpoint asks
    # for, and every family has to degenerate to the same single boundary at
    # sigma_max: linear's default spacing rule and linear_quadratic's two
    # regions both have nothing to divide, which is a degenerate input, not an
    # invalid one.
    @pytest.mark.parametrize("name", ["linear", "linear_quadratic"])
    def test_a_single_step_schedule_is_buildable(self, name: str) -> None:
        schedule = build_sigma_schedule(ScheduleConfig(name=name, steps=1))
        assert schedule.tolist()[-1] == 0.0

    def test_karras_can_build_a_single_step_schedule(self) -> None:
        # Karras derives its floor from a constant rather than from the step
        # count, so it reaches the one-step case without any special handling.
        assert build_sigma_schedule(
            ScheduleConfig(name="karras", steps=1)
        ).tolist() == [1.0, 0.0]

    def test_zero_steps_reports_the_documented_error(self) -> None:
        # Every family derives a default sigma_min from the step count, so the
        # step count has to be validated before that arithmetic runs — otherwise
        # steps=0 surfaces as ZeroDivisionError instead of the documented error.
        for build in (linear_sigmas, linear_quadratic_sigmas, karras_sigmas):
            with pytest.raises(ValueError, match="steps must be a positive integer"):
                build(0)

    @pytest.mark.parametrize(
        ("builder", "kwargs"),
        [
            (karras_sigmas, {"steps": 0}),
            (linear_sigmas, {"steps": 4, "sigma_max": 2.0}),
            (linear_sigmas, {"steps": 4, "sigma_min": 0.0}),
            (karras_sigmas, {"steps": 4, "rho": 0.0}),
            (linear_quadratic_sigmas, {"steps": 4, "linear_steps": 4}),
            (linear_quadratic_sigmas, {"steps": 4, "threshold_noise": 1.0}),
        ],
    )
    def test_families_reject_arguments_outside_their_domain(
        self, builder: Any, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            builder(**kwargs)


class TestScheduleMatchesTrainingShift:
    """The inference shift must be numerically identical to the training shift.

    ``avgen.infer.schedule`` and ``avgen.train.timestep`` each implement the
    reparametrisation and the sequence-length interpolation, because inference
    must not import the trainer. Two implementations of one formula drift, and
    the symptom of drift here is a systematically over- or under-denoised sample
    that is very hard to attribute — so the two are pinned against each other.
    """

    @pytest.mark.parametrize("shift", [0.5, 1.0, 2.0, 3.0, 7.0])
    def test_apply_shift_equals_shift_timesteps_bitwise(self, shift: float) -> None:
        timesteps = torch.rand(256)
        assert torch.equal(
            apply_shift(timesteps, shift), shift_timesteps(timesteps, shift)
        )

    @pytest.mark.parametrize("length", [1, 100, 1024, 5000, 32768, 200000])
    def test_the_interpolations_agree_at_every_regime(self, length: int) -> None:
        # Below the low anchor, between them, and past the high anchor: the
        # clamping behaviour has to agree too, not just the linear region.
        inference = resolution_shift(
            length,
            base_length=1024,
            base_shift=0.95,
            max_length=32768,
            max_shift=2.05,
        )
        training = train_resolution_shift(
            length,
            base_seq_len=1024,
            base_shift=0.95,
            max_seq_len=32768,
            max_shift=2.05,
        )
        assert inference == training

    def test_a_dynamic_schedule_reproduces_the_training_samplers_shift(self) -> None:
        # This is the end-to-end version: what the checkpoint was trained with,
        # and what the schedule builder will resolve, must be the same number.
        sampler = ShiftedLogitNormalSampler()
        config = ScheduleConfig(
            name="linear",
            steps=8,
            dynamic_shift=True,
            base_length=sampler.base_seq_len,
            base_shift=sampler.base_shift,
            max_length=sampler.max_seq_len,
            max_shift=sampler.max_shift,
        )
        for length in (256, 4096, 32768, 100000):
            schedule = build_sigma_schedule(config, sequence_length=length)
            assert schedule.shift == sampler.shift_for(length)

    def test_the_shifted_schedule_equals_the_shifted_training_density(self) -> None:
        # Not just the scalar: the curve the schedule steps through must be the
        # image of the unshifted curve under the training-time map.
        sampler = ShiftedLogitNormalSampler()
        length = 4096
        config = ScheduleConfig(
            name="linear",
            steps=10,
            dynamic_shift=True,
            base_length=sampler.base_seq_len,
            base_shift=sampler.base_shift,
            max_length=sampler.max_seq_len,
            max_shift=sampler.max_shift,
        )
        shifted = build_sigma_schedule(config, sequence_length=length).sigmas
        unshifted = linear_sigmas(10)
        expected = shift_timesteps(unshifted, sampler.shift_for(length))
        torch.testing.assert_close(shifted[:-1], expected[:-1], rtol=0, atol=1e-6)

    def test_a_dynamic_shift_without_a_length_is_an_error_not_a_default(self) -> None:
        # Falling back to 1.0 here would silently un-shift a high-resolution
        # sample, which is the exact failure the module warning describes.
        config = ScheduleConfig(name="linear", steps=4, dynamic_shift=True)
        with pytest.raises(
            ValueError, match="dynamic_shift requires a sequence_length"
        ):
            build_sigma_schedule(config)

    def test_a_static_shift_ignores_the_sequence_length(self) -> None:
        config = ScheduleConfig(name="linear", steps=4, shift=2.0)
        assert config.resolve_shift(None) == 2.0
        assert config.resolve_shift(999999) == 2.0

    def test_the_exponential_parameterisation_is_opt_in(self) -> None:
        # The Flux/SD3 mu form and the plain form disagree by a lot, so using
        # the wrong one silently mis-shifts every sample.
        plain = resolution_shift(4096, base_shift=1.0, max_shift=3.0)
        exponential = resolution_shift(
            4096, base_shift=1.0, max_shift=3.0, exponential=True
        )
        assert exponential == pytest.approx(math.exp(plain))

    def test_rejects_coincident_anchors(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            resolution_shift(512, base_length=1024, max_length=1024)


class TestSamplerRegistry:
    """A pinned generation config naming a sampler must resolve forever.

    The registry is what a serialised config resolves through, so replacement
    is refused and an unknown name reports what is available rather than
    failing with a bare KeyError at the top of a long job.
    """

    def test_every_shipped_solver_is_registered(self) -> None:
        assert set(list_samplers()) == {
            "dpmpp_2m",
            "euler",
            "euler_ancestral",
            "heun",
            "res_multistep",
        }

    def test_a_bare_name_builds_with_defaults(self) -> None:
        assert build_sampler("euler").config == SamplerConfig(name="euler")

    def test_each_build_is_a_fresh_instance(self) -> None:
        # Multistep state belongs to one generation; a shared instance would
        # extrapolate one sample from another's history.
        assert build_sampler("dpmpp_2m") is not build_sampler("dpmpp_2m")

    def test_an_unknown_name_lists_what_exists(self) -> None:
        with pytest.raises(KeyError, match="unknown sampler"):
            build_sampler("plms")

    def test_re_registering_a_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_sampler("euler")

    def test_evaluation_budget_is_declared_honestly(self) -> None:
        # Comparing solvers at equal step counts is meaningless; the budget that
        # matters is model evaluations, so it has to be readable off the solver.
        assert build_sampler("euler").evaluations_per_step == 1
        assert build_sampler("heun").evaluations_per_step == 2

    def test_survives_a_serialisation_round_trip(self) -> None:
        config = SamplerConfig(name="euler_ancestral", eta=0.4, s_noise=1.0)
        assert SamplerConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize(
        "kwargs",
        [{"name": ""}, {"eta": -0.1}, {"eta": 1.5}, {"s_noise": -1.0}],
    )
    def test_rejects_an_invalid_config(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SamplerConfig(**kwargs)


class TestSamplerCorrectness:
    """Every solver must integrate a known velocity field to the known answer.

    This is the highest-value assertion in the file. For rectified flow the
    trajectory between a clean sample and a noise draw is a straight line, so
    the exact endpoint of the integration is available in closed form and every
    solver — first order, second order, single-step, multistep, exponential
    integrator — has to land on it. A sign error, an ``alpha``/``sigma`` mixup,
    or a misplaced ``1 -`` anywhere in the update produces a sample that is
    merely worse rather than obviously broken, and nothing else would catch it.
    """

    @pytest.mark.parametrize("name", list_samplers())
    @pytest.mark.parametrize("schedule_name", list_sigma_schedules())
    def test_the_constant_field_integrates_exactly(
        self, name: str, schedule_name: str
    ) -> None:
        torch.manual_seed(0)
        schedule = build_sigma_schedule(ScheduleConfig(name=schedule_name, steps=12))
        clean = torch.randn(2, 3, 4)
        noise = torch.randn(2, 3, 4)
        start_sigma = float(schedule.sigmas[0])
        start = (1.0 - start_sigma) * clean + start_sigma * noise
        result = integrate(name, constant_field(noise - clean), start, schedule)
        error = float((result - clean).abs().max())
        assert error < SOLVER_EXACT, f"{name}/{schedule_name} drifted by {error:.3e}"

    @pytest.mark.parametrize(
        "name", ["dpmpp_2m", "euler", "euler_ancestral", "res_multistep"]
    )
    def test_the_spatially_varying_field_integrates_exactly(self, name: str) -> None:
        # constant_field cannot distinguish a solver that ignores its input from
        # one that does not. This field genuinely depends on x, and every
        # trajectory is still a straight line through the data point, so the
        # exact endpoint is known.
        torch.manual_seed(1)
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=12))
        clean = torch.randn(2, 3, 4)
        noise = torch.randn(2, 3, 4)
        start_sigma = float(schedule.sigmas[0])
        start = (1.0 - start_sigma) * clean + start_sigma * noise
        result = integrate(name, dirac_field(clean), start, schedule)
        error = float((result - clean).abs().max())
        assert error < SOLVER_EXACT, f"{name} drifted by {error:.3e}"

    def test_heun_is_exact_on_the_spatially_varying_field_away_from_the_pole(
        self,
    ) -> None:
        # Heun is excluded from the sweep above for a real reason rather than a
        # convenience one: the field has a pole at t = 0, so on the terminal step
        # its second evaluation reads a velocity of zero at an already-clean
        # probe and the trapezoid halves the step. That is a property of this
        # test field, not of Heun — on every interior step it is exact, which is
        # what this asserts.
        torch.manual_seed(2)
        clean = torch.randn(2, 3, 4)
        noise = torch.randn(2, 3, 4)
        field = dirac_field(clean)
        sigma_t, sigma_next = 0.8, 0.3
        x_t = (1.0 - sigma_t) * clean + sigma_t * noise
        solver = build_sampler("heun")
        result = solver.step(
            field(x_t, sigma_t), x_t, sigma_t, sigma_next, denoise=field
        )
        expected = clean + (sigma_next / sigma_t) * (x_t - clean)
        torch.testing.assert_close(result, expected, rtol=0, atol=SOLVER_EXACT)

    def test_a_single_euler_step_of_any_size_lands_on_the_target(self) -> None:
        # For a perfectly trained rectified flow this is not an approximation:
        # everything more elaborate in the module exists to compensate for a
        # real model's field only being approximately constant.
        torch.manual_seed(3)
        clean = torch.randn(2, 8)
        noise = torch.randn(2, 8)
        result = build_sampler("euler").step(noise - clean, noise, 1.0, 0.0)
        torch.testing.assert_close(result, clean, rtol=0, atol=1e-6)

    def test_to_x0_is_exact_for_the_flow_interpolant(self) -> None:
        # Substituting x_t = (1-s) x0 + s eps and v = eps - x0 into x_t - s v
        # cancels eps entirely. Three of the solvers work in x0 space, so this
        # conversion being exact rather than approximate is load-bearing.
        torch.manual_seed(4)
        clean, noise, sigma = torch.randn(2, 8), torch.randn(2, 8), 0.37
        x_t = (1.0 - sigma) * clean + sigma * noise
        torch.testing.assert_close(
            to_x0(x_t, noise - clean, sigma), clean, rtol=0, atol=1e-6
        )

    def test_to_epsilon_is_exact_for_the_flow_interpolant(self) -> None:
        torch.manual_seed(5)
        clean, noise, sigma = torch.randn(2, 8), torch.randn(2, 8), 0.37
        x_t = (1.0 - sigma) * clean + sigma * noise
        torch.testing.assert_close(
            to_epsilon(x_t, noise - clean, sigma), noise, rtol=0, atol=1e-6
        )

    def test_heun_refuses_to_silently_degrade_to_euler(self) -> None:
        # Falling back to one evaluation would make a heun run cost what euler
        # costs and produce what euler produces, under a config that says heun.
        with pytest.raises(ValueError, match="heun requires a denoise callback"):
            build_sampler("heun").step(torch.zeros(2, 4), torch.zeros(2, 4), 0.5, 0.2)

    @pytest.mark.parametrize("name", list_samplers())
    def test_every_solver_refuses_a_step_that_does_not_advance(self, name: str) -> None:
        solver = build_sampler(name)
        with pytest.raises(ValueError, match="must be below"):
            solver.step(torch.zeros(2, 4), torch.zeros(2, 4), 0.3, 0.3)

    @pytest.mark.parametrize("name", list_samplers())
    def test_every_solver_refuses_a_non_finite_sigma(self, name: str) -> None:
        solver = build_sampler(name)
        with pytest.raises(ValueError, match="must be finite"):
            solver.step(torch.zeros(2, 4), torch.zeros(2, 4), math.nan, 0.1)

    @pytest.mark.parametrize("name", ["dpmpp_2m", "res_multistep"])
    def test_reset_discards_multistep_history(self, name: str) -> None:
        # Without it, the first step of a new generation extrapolates from the
        # previous generation's clean-sample prediction.
        torch.manual_seed(6)
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=6))
        clean, noise = torch.randn(2, 8), torch.randn(2, 8)
        field = constant_field(noise - clean)
        solver = build_sampler(name)

        def run() -> torch.Tensor:
            solver.reset()
            current = noise
            for _, sigma, sigma_next in schedule:
                current = solver.step(field(current, sigma), current, sigma, sigma_next)
            return current

        assert torch.equal(run(), run())


class TestEulerAncestral:
    """The ancestral solver is the ODE-to-SDE conversion the RL subsystem needs.

    Policy-gradient methods over diffusion need the log-probability of the
    transition actually taken, which is only defined if the noise injection is
    visible rather than buried in a scheduler. And ``eta = 0`` must reduce to
    Euler *exactly*, because that equivalence is what lets an RL run share one
    code path with the deterministic evaluation it is measured against.
    """

    def test_eta_zero_is_bit_identical_to_euler(self) -> None:
        torch.manual_seed(7)
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=9))
        start = torch.randn(2, 3, 4)
        velocity = torch.randn(2, 3, 4)
        euler = build_sampler("euler")
        ancestral = build_sampler(SamplerConfig(name="euler_ancestral", eta=0.0))
        left, right = start, start
        for _, sigma, sigma_next in schedule:
            left = euler.step(velocity, left, sigma, sigma_next)
            right = ancestral.step(velocity, right, sigma, sigma_next)
        assert torch.equal(left, right), "eta=0 must not merely approximate euler"

    def test_eta_zero_reports_itself_deterministic(self) -> None:
        assert not build_sampler(
            SamplerConfig(name="euler_ancestral", eta=0.0)
        ).is_stochastic
        assert build_sampler(
            SamplerConfig(name="euler_ancestral", eta=0.5)
        ).is_stochastic

    def test_the_split_recombines_to_the_target_noise_level(self) -> None:
        # sigma_down and sigma_up are the two legs of one variance budget; if
        # they did not recombine to sigma_next the sample would arrive at the
        # wrong noise level and the next step would be solving a different ODE.
        solver = EulerAncestralSampler(SamplerConfig(name="euler_ancestral", eta=0.7))
        sigma_down, sigma_up = solver.ancestral_split(0.8, 0.5)
        assert sigma_down**2 + sigma_up**2 == pytest.approx(0.5**2)

    def test_no_noise_is_injectable_into_a_clean_target(self) -> None:
        # The sample must be exactly clean at the end of the schedule.
        solver = EulerAncestralSampler(SamplerConfig(name="euler_ancestral", eta=1.0))
        assert solver.ancestral_split(0.1, 0.0) == (0.0, 0.0)

    def test_a_stochastic_step_refuses_the_global_rng(self) -> None:
        # Falling back to the ambient RNG would break seed reproducibility while
        # still producing plausible output, which is the worst combination.
        solver = build_sampler(SamplerConfig(name="euler_ancestral", eta=0.5))
        with pytest.raises(ValueError, match=r"needs a torch\.Generator"):
            solver.step(torch.zeros(2, 4), torch.zeros(2, 4), 0.5, 0.2, generator=None)

    def test_replaying_a_recorded_noise_draw_reproduces_the_transition(self) -> None:
        # This is how an RL trainer recomputes the log-probability of a
        # trajectory it already sampled, under updated parameters.
        torch.manual_seed(8)
        solver = EulerAncestralSampler(SamplerConfig(name="euler_ancestral", eta=0.6))
        x_t, velocity = torch.randn(2, 5), torch.randn(2, 5)
        generator = torch.Generator().manual_seed(21)
        first, noise, sigma_up = solver.step_with_noise(
            velocity, x_t, 0.8, 0.5, generator=generator
        )
        assert sigma_up > 0.0
        replayed, replay_noise, replay_up = solver.step_with_noise(
            velocity, x_t, 0.8, 0.5, noise=noise
        )
        assert torch.equal(replayed, first)
        assert torch.equal(replay_noise, noise)
        assert replay_up == sigma_up

    def test_a_deterministic_step_reports_zero_injected_noise(self) -> None:
        solver = EulerAncestralSampler(SamplerConfig(name="euler_ancestral", eta=0.0))
        _, noise, sigma_up = solver.step_with_noise(
            torch.zeros(2, 5), torch.zeros(2, 5), 0.8, 0.5
        )
        assert sigma_up == 0.0
        assert float(noise.abs().max()) == 0.0

    def test_the_same_seed_gives_the_same_trajectory_and_a_different_one_does_not(
        self,
    ) -> None:
        torch.manual_seed(9)
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=6))
        clean, noise = torch.randn(2, 6), torch.randn(2, 6)
        field = constant_field(noise - clean)
        first = integrate("euler_ancestral", field, noise, schedule, eta=0.8, seed=3)
        same = integrate("euler_ancestral", field, noise, schedule, eta=0.8, seed=3)
        other = integrate("euler_ancestral", field, noise, schedule, eta=0.8, seed=4)
        assert torch.equal(first, same)
        assert not torch.equal(first, other)


class TestGuidance:
    """Guidance is the most effective knob and the most reliable source of artifacts.

    Three properties matter enough to pin. Scale 1.0 must be *bit-identical* to
    unguided, or "turn guidance off" is not a thing a caller can do without
    changing the sample. CFG-rescale must actually restore the conditional
    prediction's magnitude, since over-saturation is a magnitude failure. And
    APG must inflate the norm less than plain extrapolation at the same scale,
    because that is its entire claim: it removes the cause of over-saturation
    rather than correcting the symptom afterwards.
    """

    def test_scale_one_returns_the_conditional_tensor_itself(self) -> None:
        # Identity, not equality: computing uncond + 1 * delta would be equal in
        # real arithmetic and differ in the last ulp, which makes the "guidance
        # off" path untestable.
        conditional = torch.randn(2, 4, 5)
        unconditional = torch.randn(2, 4, 5)
        assert classifier_free_guidance(conditional, unconditional, 1.0) is conditional
        assert (
            adaptive_projected_guidance(conditional, unconditional, 1.0) is conditional
        )
        assert (
            apply_guidance(conditional, unconditional, config=GuidanceConfig(scale=1.0))
            is conditional
        )

    def test_an_unconfigured_call_is_unguided(self) -> None:
        conditional = torch.randn(2, 4)
        assert apply_guidance(conditional, None, config=None) is conditional

    def test_guidance_extrapolates_along_the_conditioning_direction(self) -> None:
        conditional = torch.randn(2, 4, 5)
        unconditional = torch.randn(2, 4, 5)
        guided = classifier_free_guidance(conditional, unconditional, 3.0)
        expected = unconditional + 3.0 * (conditional - unconditional)
        torch.testing.assert_close(guided, expected, rtol=0, atol=1e-6)

    def test_rescale_restores_the_conditionals_standard_deviation(self) -> None:
        # The failure it repairs: guidance extrapolates, so std(guided) grows
        # with the scale while std(cond) does not, and the implied clean sample
        # lands outside the range the decoder was trained on.
        torch.manual_seed(10)
        conditional = torch.randn(3, 4, 5)
        unconditional = torch.randn(3, 4, 5)
        guided = classifier_free_guidance(conditional, unconditional, 8.0)
        rescaled = rescale_guidance(guided, conditional, 1.0)
        dims = (1, 2)
        torch.testing.assert_close(
            rescaled.std(dim=dims), conditional.std(dim=dims), rtol=1e-5, atol=1e-5
        )

    def test_rescale_statistics_are_per_sample_not_per_batch(self) -> None:
        # Two prompts in one batch have no reason to share a magnitude; coupling
        # them would make a sample depend on what it was batched with.
        torch.manual_seed(11)
        conditional = torch.stack([torch.randn(6) * 0.1, torch.randn(6) * 10.0])
        unconditional = torch.randn(2, 6)
        guided = classifier_free_guidance(conditional, unconditional, 5.0)
        rescaled = rescale_guidance(guided, conditional, 1.0)
        torch.testing.assert_close(
            rescaled.std(dim=1), conditional.std(dim=1), rtol=1e-4, atol=1e-6
        )

    def test_rescale_at_zero_is_a_no_op(self) -> None:
        guided = torch.randn(2, 4)
        assert rescale_guidance(guided, torch.randn(2, 4), 0.0) is guided

    def test_apg_inflates_the_norm_less_than_plain_cfg(self) -> None:
        # The parallel component of the update points along what the model
        # already predicts, so amplifying it only scales the prediction up. APG
        # discards it, which is why a scale of 15 can behave like 15 on content
        # without behaving like 15 on contrast.
        torch.manual_seed(12)
        conditional = torch.randn(4, 3, 8)
        unconditional = torch.randn(4, 3, 8)
        plain = classifier_free_guidance(conditional, unconditional, 10.0)
        projected = adaptive_projected_guidance(
            conditional, unconditional, 10.0, eta=0.0
        )
        plain_norm = plain.flatten(1).norm(dim=1)
        projected_norm = projected.flatten(1).norm(dim=1)
        assert bool((projected_norm < plain_norm).all()), (
            f"APG norms {projected_norm.tolist()} are not below CFG's "
            f"{plain_norm.tolist()}"
        )

    def test_the_apg_update_is_orthogonal_to_the_conditional_prediction(self) -> None:
        torch.manual_seed(13)
        conditional = torch.randn(4, 3, 8)
        unconditional = torch.randn(4, 3, 8)
        projected = adaptive_projected_guidance(
            conditional, unconditional, 9.0, eta=0.0
        )
        update = (projected - conditional).flatten(1)
        overlap = (update * conditional.flatten(1)).sum(dim=1)
        # Relative to the magnitudes involved this is float32 noise, not a
        # residual parallel component.
        scale = update.norm(dim=1) * conditional.flatten(1).norm(dim=1)
        assert bool((overlap.abs() / scale < 1e-5).all())

    def test_the_apg_norm_threshold_bounds_one_outlier_step(self) -> None:
        # A single timestep occasionally produces a difference far larger than
        # its neighbours; bounding it stops one step dominating the trajectory.
        torch.manual_seed(14)
        conditional = torch.randn(2, 16)
        unconditional = conditional - torch.randn(2, 16) * 100.0
        clamped = adaptive_projected_guidance(
            conditional, unconditional, 5.0, norm_threshold=1.0
        )
        unclamped = adaptive_projected_guidance(conditional, unconditional, 5.0)
        assert float((clamped - conditional).norm()) < float(
            (unclamped - conditional).norm()
        )

    def test_guidance_branches_must_agree_in_shape(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            classifier_free_guidance(torch.randn(2, 4), torch.randn(2, 5), 3.0)

    def test_enabled_guidance_without_a_null_branch_is_an_error(self) -> None:
        # A silent fallback to the conditional would make a scale of 8 do
        # nothing, at full cost, with no message.
        with pytest.raises(ValueError, match="no unconditional prediction"):
            apply_guidance(torch.randn(2, 4), None, config=GuidanceConfig(scale=8.0))

    def test_scale_one_with_no_modality_override_disables_the_null_pass(self) -> None:
        assert not GuidanceConfig(scale=1.0).is_enabled
        assert GuidanceConfig(scale=1.0, text_scale=1.5).is_enabled
        assert GuidanceConfig(scale=2.0).is_enabled

    @pytest.mark.parametrize("schedule", ["constant", "linear", "cosine", "power"])
    def test_every_schedule_starts_at_the_base_scale(self, schedule: str) -> None:
        config = GuidanceConfig(scale=6.0, min_scale=1.0, schedule=schedule)
        assert config.scale_at(0.0) == pytest.approx(6.0)

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "power"])
    def test_decaying_schedules_reach_the_floor_at_the_clean_end(
        self, schedule: str
    ) -> None:
        # The late steps are where a large scale over-saturates, since the
        # predictions are confident and the extrapolation runs furthest
        # off-manifold.
        config = GuidanceConfig(scale=6.0, min_scale=1.0, schedule=schedule)
        assert config.scale_at(1.0) == pytest.approx(1.0)

    def test_the_interval_schedule_guides_only_inside_its_window(self) -> None:
        config = GuidanceConfig(
            scale=6.0,
            min_scale=1.0,
            schedule="interval",
            start_fraction=0.2,
            end_fraction=0.8,
        )
        assert config.scale_at(0.1) == 1.0
        assert config.scale_at(0.5) == 6.0
        assert config.scale_at(0.9) == 1.0

    def test_modality_factors_compose_multiplicatively(self) -> None:
        # Halving the base scale must halve everything uniformly and leave the
        # relative balance between prompt and anchor untouched, which an
        # additive composition would not.
        config = GuidanceConfig(scale=4.0, text_scale=1.5, audio_scale=0.5)
        assert config.modality_scale("text") == pytest.approx(6.0)
        assert config.modality_scale("audio") == pytest.approx(2.0)
        assert config.modality_scale("video") == pytest.approx(4.0)

    def test_an_unknown_modality_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown guidance modality"):
            GuidanceConfig(scale=2.0).modality_scale("depth")

    def test_progress_outside_the_trajectory_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"progress must be in \[0, 1\]"):
            GuidanceConfig(scale=2.0, schedule="linear").scale_at(1.5)

    def test_a_modality_chain_sums_marginal_increments(self) -> None:
        # Chain form rather than independent differences from a shared null:
        # each increment measures the marginal effect of that conditioning given
        # the ones already applied, which is what an independent knob controls.
        torch.manual_seed(15)
        base, with_text, with_video = (
            torch.randn(3, 2, 4),
            torch.randn(3, 2, 4),
            torch.randn(3, 2, 4),
        )
        config = GuidanceConfig(scale=2.0, text_scale=1.5, video_scale=0.5)
        composed = apply_modality_guidance(
            [("base", base), ("text", with_text), ("video", with_video)], config=config
        )
        expected = base + 3.0 * (with_text - base) + 1.0 * (with_video - with_text)
        torch.testing.assert_close(composed, expected, rtol=0, atol=1e-5)

    def test_a_chain_needs_at_least_a_base_and_one_branch(self) -> None:
        with pytest.raises(ValueError, match="at least a base"):
            apply_modality_guidance(
                [("text", torch.randn(2, 4))], config=GuidanceConfig()
            )

    def test_survives_a_serialisation_round_trip(self) -> None:
        config = GuidanceConfig(
            scale=6.0, text_scale=1.5, rescale=0.7, projection=True, schedule="cosine"
        )
        assert GuidanceConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale": 0.0},
            {"scale": math.inf},
            {"text_scale": -1.0},
            {"rescale": 1.5},
            {"projection_threshold": -1.0},
            {"schedule": "sigmoid"},
            {"schedule_power": 0.0},
            {"start_fraction": 0.8, "end_fraction": 0.2},
            {"end_fraction": 1.5},
        ],
    )
    def test_rejects_an_invalid_config(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            GuidanceConfig(**kwargs)


# ---------------------------------------------------------------------------
# Conditioning: the train/inference identity


VIDEO_FPS = 8.0
AUDIO_FPS = 12.0
SIGMA = 0.6


class _CapturingModel(nn.Module):
    """Records the ModelInput the training objective builds, and returns zeros.

    The objective assembles its input and hands it straight to the model, so
    this is the only way to observe the training-time tensor layout without
    reimplementing the assembly in the test — which would defeat the point.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen: ModelInput | None = None

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.seen = inputs
        return ModelOutput(
            video=torch.zeros_like(inputs.video.tokens),
            audio=torch.zeros_like(inputs.audio.tokens),
        )


class _FixedConditioning:
    """A conditioning sampler that always draws one chosen plan.

    The shipped sampler draws a task mixture from an RNG; pinning it makes the
    training-side input a deterministic function of the batch, which is what
    lets the inference-side input be compared against it tensor by tensor.
    """

    def __init__(
        self,
        mode: ConditionMode,
        *,
        video_prefix: int = 0,
        video_all: bool = False,
        audio_all: bool = False,
    ) -> None:
        self.mode = mode
        self.video_prefix = video_prefix
        self.video_all = video_all
        self.audio_all = audio_all

    def sample(self, batch: MediaBatch, *, rng: RNGStreams) -> ConditioningPlan:
        del rng
        size = batch.spec.batch_size
        _, _, frames, height, width = batch.spec.video_shape
        video = torch.zeros((size, frames, height, width), dtype=torch.bool)
        if self.video_all:
            video[:] = True
        elif self.video_prefix:
            video[:, : self.video_prefix] = True
        audio = torch.zeros((size, batch.spec.audio_tokens), dtype=torch.bool)
        if self.audio_all:
            audio[:] = True
        return ConditioningPlan(
            condition_mode=torch.full((size,), int(self.mode), dtype=torch.int64),
            video_conditioned=video,
            audio_conditioned=audio,
            drop_text=torch.zeros((size,), dtype=torch.bool),
        )


class _FixedTimestep:
    """A timestep sampler that always returns one noise level."""

    def sample(
        self,
        batch: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        del generator, sequence_length
        return torch.full((batch,), SIGMA, dtype=torch.float32, device=device)


def make_batch(*, audio_frames: int = 0, seed: int = 0) -> MediaBatch:
    """Return a tiny clean batch with the coordinates a StreamState reproduces."""
    torch.manual_seed(seed)
    batch, channels, frames, height, width = 2, 4, 4, 4, 4
    spec = MediaBatchSpec(
        schema_version=1,
        bucket_id=0,
        video_shape=(batch, channels, frames, height, width),
        audio_shape=(batch, 4, audio_frames),
        text_shape=(batch, 6, 32),
        video_timebase_num=int(VIDEO_FPS),
        video_timebase_den=1,
        audio_timebase_num=int(AUDIO_FPS),
        audio_timebase_den=1,
        video_codec_id="reference-video-v1",
        audio_codec_id="reference-audio-v1",
    )
    media = MediaBatch(
        video=torch.randn(spec.video_shape),
        audio=torch.randn(spec.audio_shape),
        text=torch.randn(spec.text_shape),
        video_mask=torch.ones((batch, frames, height, width), dtype=torch.bool),
        audio_mask=torch.ones((batch, audio_frames), dtype=torch.bool),
        video_positions=torch.arange(frames, dtype=torch.float32).repeat(batch, 1)
        / VIDEO_FPS,
        audio_positions=torch.arange(audio_frames, dtype=torch.float32).repeat(batch, 1)
        / AUDIO_FPS,
        sample_ids=torch.arange(batch, dtype=torch.int64),
        spec=spec,
    )
    media.validate()
    return media


def training_input(
    batch: MediaBatch,
    conditioning: _FixedConditioning,
    patchifier: GridPatchifier,
) -> ModelInput:
    """Run the real training objective and return the input it built."""
    objective = FlowMatchingObjective(
        FlowMatchingConfig(validate=True),
        _FixedTimestep(),
        conditioning,
        audio_patchifier=default_audio_patchifier(),
    )
    model = _CapturingModel()
    objective(model, batch, RNGStreams.from_seed(17), patchifier=patchifier)
    assert model.seen is not None
    return model.seen


def assert_streams_identical(
    training: ModelInput, inference: ModelInput, *, stream: str
) -> None:
    """Assert that one modality's token stream matches field by field."""
    left = getattr(training, stream)
    right = getattr(inference, stream)
    assert left.layout == right.layout
    for field in ("tokens", "coords", "mask", "conditioned"):
        assert torch.equal(getattr(left, field), getattr(right, field)), (
            f"{stream}.{field} differs between the training and inference paths"
        )
    # Compared through expanded_noise rather than the raw field: an unanchored
    # stream legitimately uses the (batch,) fast path on one side and the
    # materialised (batch, tokens) form on the other, and the values are what
    # the model reads.
    assert torch.equal(left.expanded_noise(), right.expanded_noise())


def text_context(batch: MediaBatch) -> TextContext:
    """Return the text context matching a batch's text features."""
    return TextContext(
        features=batch.text,
        mask=torch.ones(batch.text.shape[:2], dtype=torch.bool),
    )


class TestConditioningMatchesTraining:
    """The inference builders must produce the training-time tensor layout.

    This is the test the "no train/inference skew" claim rests on. The failure
    it prevents is specific: an inference path that anchors by *overwriting the
    prediction* after each step teaches the model nothing about the anchor
    during the step, so the model denoises the anchored region as if it were
    noise, the overwrite hides the mismatch at the region boundary, and the
    result is a video whose first frame is correct and whose second frame is
    unrelated to it.

    Each test drives the *real* training objective, unfolds the noisy grid it
    produced, feeds that grid back through the inference builder, and requires
    the two streams to be bit-identical.
    """

    def test_image_to_video_reproduces_the_training_layout_exactly(self) -> None:
        prefix = 2
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        batch = make_batch()
        training = training_input(
            batch,
            _FixedConditioning(ConditionMode.IMAGE_TO_VIDEO, video_prefix=prefix),
            patchifier,
        )
        # The sampler hands the inference path a noisy grid; unfolding the
        # training stream gives exactly the grid a correct sampler would hold.
        state = StreamState(latents=patchifier.to_grid(training.video), fps=VIDEO_FPS)
        inference = condition_first_frame(
            state,
            batch.video[:, :, :prefix],
            sigma=SIGMA,
            text=text_context(batch),
            patchifier=patchifier,
            latent_frames=prefix,
        )
        assert_streams_identical(training, inference, stream="video")
        assert torch.equal(training.condition_mode, inference.condition_mode)

    def test_continuation_reproduces_the_training_layout_exactly(self) -> None:
        prefix = 3
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        batch = make_batch(seed=1)
        training = training_input(
            batch,
            _FixedConditioning(ConditionMode.CONTINUATION, video_prefix=prefix),
            patchifier,
        )
        state = StreamState(latents=patchifier.to_grid(training.video), fps=VIDEO_FPS)
        inference = condition_temporal_prefix(
            state,
            batch.video[:, :, :prefix],
            sigma=SIGMA,
            text=text_context(batch),
            patchifier=patchifier,
        )
        assert_streams_identical(training, inference, stream="video")
        assert torch.equal(training.condition_mode, inference.condition_mode)

    def test_inpainting_reproduces_the_training_layout_exactly(self) -> None:
        # A patch-aligned mask, because the patchifier reduces the anchor mask
        # with `all`: a straddling patch is correctly treated as noised on the
        # training side and would legitimately differ here.
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        batch = make_batch(seed=2)
        training = training_input(
            batch, _FixedConditioning(ConditionMode.INPAINT, video_prefix=2), patchifier
        )
        state = StreamState(latents=patchifier.to_grid(training.video), fps=VIDEO_FPS)
        mask = torch.zeros((2, 4, 4, 4), dtype=torch.bool)
        mask[:, :2] = True
        inference = condition_mask(
            state,
            batch.video,
            mask,
            sigma=SIGMA,
            text=text_context(batch),
            patchifier=patchifier,
        )
        assert_streams_identical(training, inference, stream="video")
        assert int(inference.condition_mode[0]) == int(ConditionMode.INPAINT)

    def test_video_to_audio_reproduces_the_training_layout_exactly(self) -> None:
        # The cross-modal case is the one where the anchored stream's noise
        # level has to be exactly zero. A stream at 1e-3 instead of 0 is one the
        # model was never shown as a conditioning signal, and cross-modal
        # alignment degrades sharply.
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        audio_patchifier = default_audio_patchifier()
        batch = make_batch(audio_frames=6, seed=3)
        training = training_input(
            batch,
            _FixedConditioning(ConditionMode.VIDEO_TO_AUDIO, video_all=True),
            patchifier,
        )
        # Audio uses unit patches, so its token axis is the latent frame axis.
        noisy_audio = training.audio.tokens.transpose(1, 2).reshape(2, 4, 6)
        inference = condition_stream(
            video=StreamState(latents=batch.video, fps=VIDEO_FPS),
            audio=StreamState(latents=noisy_audio, fps=AUDIO_FPS),
            generate="audio",
            sigma=SIGMA,
            text=text_context(batch),
            patchifier=patchifier,
            audio_patchifier=audio_patchifier,
        )
        assert_streams_identical(training, inference, stream="video")
        assert_streams_identical(training, inference, stream="audio")
        assert torch.equal(training.condition_mode, inference.condition_mode)

    def test_audio_to_video_reproduces_the_training_layout_exactly(self) -> None:
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        batch = make_batch(audio_frames=6, seed=4)
        training = training_input(
            batch,
            _FixedConditioning(ConditionMode.AUDIO_TO_VIDEO, audio_all=True),
            patchifier,
        )
        inference = condition_stream(
            video=StreamState(
                latents=patchifier.to_grid(training.video), fps=VIDEO_FPS
            ),
            audio=StreamState(latents=batch.audio, fps=AUDIO_FPS),
            generate="video",
            sigma=SIGMA,
            text=text_context(batch),
            patchifier=patchifier,
            audio_patchifier=default_audio_patchifier(),
        )
        assert_streams_identical(training, inference, stream="video")
        assert_streams_identical(training, inference, stream="audio")
        assert torch.equal(training.condition_mode, inference.condition_mode)

    def test_the_unanchored_task_reproduces_the_training_layout(self) -> None:
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        batch = make_batch(seed=5)
        training = training_input(
            batch, _FixedConditioning(ConditionMode.VIDEO_ONLY), patchifier
        )
        state = StreamState(latents=patchifier.to_grid(training.video), fps=VIDEO_FPS)
        inference = unconditional_input(
            state, sigma=SIGMA, text=text_context(batch), patchifier=patchifier
        )
        assert_streams_identical(training, inference, stream="video")

    def test_the_unanchored_fast_path_uses_a_per_sample_noise_level(self) -> None:
        # Documenting a real divergence in *layout* rather than value: the
        # inference builder takes the (batch,) fast path when nothing is
        # anchored, while the training objective always materialises the
        # per-token form. expanded_noise() agrees, so the model reads the same
        # numbers, but the streams are not field-for-field identical and the
        # module docstring's claim that the fast path is "exactly what training
        # does for the unanchored tasks" is not literally true.
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        batch = make_batch(seed=6)
        training = training_input(
            batch, _FixedConditioning(ConditionMode.VIDEO_ONLY), patchifier
        )
        state = StreamState(latents=patchifier.to_grid(training.video), fps=VIDEO_FPS)
        inference = unconditional_input(
            state, sigma=SIGMA, text=text_context(batch), patchifier=patchifier
        )
        assert training.video.noise_level.ndim == 2
        assert inference.video.noise_level.ndim == 1
        assert torch.equal(
            training.video.expanded_noise(), inference.video.expanded_noise()
        )

    @pytest.mark.parametrize(
        "task",
        ["first_frame", "prefix", "mask", "video_to_audio", "audio_to_video"],
    )
    def test_every_anchored_task_puts_exactly_zero_noise_on_its_anchors(
        self, task: str
    ) -> None:
        # The model must *know* which regions are clean from the noise level it
        # is handed, rather than having to infer it from their statistics.
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        text = TextContext.empty(2, 8)
        video = StreamState(latents=torch.randn(2, 4, 4, 4, 4), fps=VIDEO_FPS)
        audio = StreamState(latents=torch.randn(2, 4, 6), fps=AUDIO_FPS)
        mask = torch.zeros((2, 4, 4, 4), dtype=torch.bool)
        mask[:, :2] = True
        builders = {
            "first_frame": lambda: condition_first_frame(
                video,
                torch.randn(2, 4, 4, 4),
                sigma=SIGMA,
                text=text,
                patchifier=patchifier,
            ),
            "prefix": lambda: condition_temporal_prefix(
                video,
                torch.randn(2, 4, 2, 4, 4),
                sigma=SIGMA,
                text=text,
                patchifier=patchifier,
            ),
            "mask": lambda: condition_mask(
                video,
                torch.randn(2, 4, 4, 4, 4),
                mask,
                sigma=SIGMA,
                text=text,
                patchifier=patchifier,
            ),
            "video_to_audio": lambda: condition_stream(
                video=video,
                audio=audio,
                generate="audio",
                sigma=SIGMA,
                text=text,
                patchifier=patchifier,
            ),
            "audio_to_video": lambda: condition_stream(
                video=video,
                audio=audio,
                generate="video",
                sigma=SIGMA,
                text=text,
                patchifier=patchifier,
            ),
        }
        inputs = builders[task]()
        for stream in (inputs.video, inputs.audio):
            if stream.length == 0:
                continue
            anchored = stream.conditioned
            noise = stream.expanded_noise()
            if bool(anchored.any()):
                assert float(noise[anchored].abs().max()) == 0.0, (
                    f"{task}: an anchored token carries a non-zero noise level"
                )
            if bool((~anchored).any()):
                assert float(noise[~anchored].min()) > 0.0, (
                    f"{task}: a free token carries a zero noise level"
                )

    def test_the_anchor_value_reaches_the_model_not_a_flag(self) -> None:
        # The model must receive the clean latent at anchored positions, not the
        # noisy one with a flag alongside it: every published conditioning
        # scheme that works feeds the clean value directly.
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        video = StreamState(latents=torch.randn(2, 4, 4, 4, 4), fps=VIDEO_FPS)
        anchor = torch.randn(2, 4, 1, 4, 4)
        inputs = condition_first_frame(
            video,
            anchor,
            sigma=SIGMA,
            text=TextContext.empty(2, 8),
            patchifier=patchifier,
        )
        grid = patchifier.to_grid(inputs.video)
        torch.testing.assert_close(grid[:, :, :1], anchor, rtol=0, atol=0)
        # The free frames must be left alone: mixing must replace the anchored
        # region and nothing else.
        torch.testing.assert_close(
            grid[:, :, 1:], video.latents[:, :, 1:], rtol=0, atol=0
        )

    def test_unconditional_input_is_not_the_guidance_null_branch(self) -> None:
        # Confusing the two produces a guidance difference that measures "with
        # versus without an anchor" instead of "with versus without a prompt".
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        text = TextContext(
            features=torch.randn(2, 6, 32), mask=torch.ones(2, 6, dtype=torch.bool)
        )
        video = StreamState(
            latents=torch.randn(2, 4, 4, 4, 4),
            fps=VIDEO_FPS,
            anchor=torch.randn(2, 4, 4, 4, 4),
            anchor_mask=torch.ones(2, 4, 4, 4, dtype=torch.bool),
        )
        inputs = unconditional_input(
            video, sigma=SIGMA, text=text, patchifier=patchifier
        )
        assert not bool(inputs.video.conditioned.any()), "anchors must be dropped"
        assert float(inputs.text.features.abs().max()) > 0.0, "text must be kept"
        # The guidance null branch does the opposite.
        assert float(inputs.unconditional().text.features.abs().max()) == 0.0

    def test_the_audio_grid_is_lifted_with_unit_spatial_extent(self) -> None:
        # One patchifier code path serves both modalities, which is why the
        # coordinate convention and the mask reduction cannot drift apart.
        audio = StreamState(latents=torch.randn(2, 4, 6), fps=AUDIO_FPS)
        assert audio.is_audio
        assert tuple(audio.grid().shape) == (2, 4, 6, 1, 1)
        assert audio.element_shape == (2, 6)

    def test_positions_are_physical_seconds_from_the_stream_start(self) -> None:
        # Non-zero start_seconds is what lets a continuation's coordinates carry
        # on from where the prefix ended rather than restarting at zero.
        state = StreamState(
            latents=torch.randn(2, 4, 3, 2, 2), fps=4.0, start_seconds=1.5
        )
        torch.testing.assert_close(
            state.positions()[0],
            torch.tensor([1.5, 1.75, 2.0]),
            rtol=0,
            atol=1e-6,
        )

    def test_first_frame_mask_covers_only_the_leading_frames(self) -> None:
        state = StreamState(latents=torch.randn(2, 4, 5, 2, 2), fps=4.0)
        mask = first_frame_mask(state, frames=2)
        assert tuple(mask.shape) == (2, 5, 2, 2)
        assert bool(mask[:, :2].all())
        assert not bool(mask[:, 2:].any())

    @pytest.mark.parametrize("frames", [0, 6, -1])
    def test_an_anchor_longer_than_the_canvas_is_rejected(self, frames: int) -> None:
        state = StreamState(latents=torch.randn(2, 4, 5, 2, 2), fps=4.0)
        with pytest.raises(ValueError, match=r"frames must be in \[1, 5\]"):
            first_frame_mask(state, frames=frames)

    @pytest.mark.parametrize("sigma", [-0.1, 1.5, math.nan])
    def test_a_flow_time_outside_the_unit_interval_is_rejected(
        self, sigma: float
    ) -> None:
        # A variance-exploding sigma reaching a flow model would produce noise,
        # so it is rejected with a message that says which convention is which.
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        video = StreamState(latents=torch.randn(2, 4, 4, 4, 4), fps=VIDEO_FPS)
        with pytest.raises(ValueError, match="flow time"):
            build_model_input(
                video=video,
                sigma=sigma,
                text=TextContext.empty(2, 8),
                patchifier=patchifier,
            )

    def test_mismatched_batch_sizes_across_modalities_are_rejected(self) -> None:
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        with pytest.raises(ValueError, match="audio batch size"):
            build_model_input(
                video=StreamState(latents=torch.randn(2, 4, 4, 4, 4), fps=VIDEO_FPS),
                sigma=SIGMA,
                text=TextContext.empty(2, 8),
                patchifier=patchifier,
                audio=StreamState(latents=torch.randn(3, 4, 6), fps=AUDIO_FPS),
            )

    def test_condition_stream_rejects_an_unknown_target(self) -> None:
        patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
        with pytest.raises(ValueError, match="generate must be"):
            condition_stream(
                video=StreamState(latents=torch.randn(2, 4, 4, 4, 4), fps=VIDEO_FPS),
                audio=StreamState(latents=torch.randn(2, 4, 6), fps=AUDIO_FPS),
                generate="text",
                sigma=SIGMA,
                text=TextContext.empty(2, 8),
                patchifier=patchifier,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fps": 0.0},
            {"fps": math.inf},
            {"start_seconds": math.nan},
            {"anchor": torch.randn(2, 4, 4, 4, 4)},
        ],
    )
    def test_stream_state_rejects_an_inconsistent_construction(
        self, kwargs: dict[str, Any]
    ) -> None:
        fields: dict[str, Any] = {
            "latents": torch.randn(2, 4, 4, 4, 4),
            "fps": VIDEO_FPS,
        }
        fields.update(kwargs)
        with pytest.raises(ValueError):
            StreamState(**fields)

    def test_stream_state_rejects_an_unsupported_rank(self) -> None:
        with pytest.raises(ValueError, match="rank-5 video grid or a rank-3"):
            StreamState(latents=torch.randn(2, 4, 4, 4), fps=VIDEO_FPS)


# ---------------------------------------------------------------------------
# The pipeline


def build_denoiser(*, seed: int = 0, responsive: bool = True) -> VideoDiT:
    """Return a tiny real VideoDiT, optionally made sensitive to its prompt.

    A freshly initialised DiT is *deliberately* an exact no-op: the output head
    and every cross-attention output projection are zero-initialised, so the
    model predicts zero and its prediction does not depend on the text at all.
    That is correct for training — the branch starts as the identity — but it
    makes every guidance assertion vacuous, so the tests that need guidance to
    do something fill those two projections with small random weights first.
    """
    torch.manual_seed(seed)
    model = VideoDiT(preset("tiny"))
    model.init_weights()
    model.eval()
    if responsive:
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if (
                    name.endswith("final_proj.weight")
                    or "cross_attention.out_proj" in name
                ):
                    parameter.copy_(
                        torch.randn(parameter.shape, generator=generator) * 0.05
                    )
    return model


def build_pipeline(
    *, model: VideoDiT | None = None, steps: int = 3, **overrides: Any
) -> GenerationPipeline:
    """Assemble a complete CPU pipeline over the reference codecs.

    ``ReferenceVideoCodec(channels=1, spatial_compression=2)`` emits exactly the
    four latent channels the ``tiny`` preset expects, so the model's patchifier
    and the codec's geometry line up without either being adjusted for the test.
    """
    config = GenerationConfig(
        steps=steps,
        seed=1,
        height=8,
        width=8,
        num_frames=4,
        fps=8.0,
        schedule=ScheduleConfig(name="linear", steps=steps),
        **overrides,
    )
    return GenerationPipeline(
        model if model is not None else build_denoiser(),
        video_codec=ReferenceVideoCodec(
            channels=1, temporal_compression=1, spatial_compression=2
        ),
        text_encoder=ReferenceTextEncoder(width=32, max_length=8),
        audio_codec=ReferenceAudioCodec(channels=1, hop_length=16, sample_rate=1600),
        config=config,
        device=torch.device("cpu"),
    )


class TestGenerationPipeline:
    """The pipeline must sequence encode, sample, and decode without surprises.

    Driven with the real ``VideoDiT`` rather than a stub, because the properties
    worth checking here — that the latent geometry the codec reports is the
    geometry the model's patchifier accepts, that the token layout survives the
    round trip through the sampler — are exactly the ones a stub would paper
    over.

    The pinned config in the result is not decoration: a seed alone reproduces
    nothing six months later if the shift, the sampler, or the guidance schedule
    has moved in the meantime, and those are the fields people edit while
    iterating.
    """

    def test_generates_media_at_the_requested_geometry(self) -> None:
        media = build_pipeline()(["a red balloon", "a blue kite"])
        assert tuple(media.video.shape) == (2, 1, 4, 8, 8)
        assert bool(torch.isfinite(media.video).all())
        assert len(media) == 2
        assert media.duration_seconds == pytest.approx(0.5)

    def test_a_bare_string_is_a_batch_of_one(self) -> None:
        media = build_pipeline()("one prompt")
        assert media.batch_size == 1
        assert media.prompts == ("one prompt",)

    def test_an_empty_prompt_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            build_pipeline()([])

    def test_the_same_seed_is_bit_identical(self) -> None:
        pipeline = build_pipeline()
        first = pipeline(["a red balloon"])
        second = pipeline(["a red balloon"])
        assert torch.equal(first.video, second.video)

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        pipeline = build_pipeline()
        first = pipeline(["a red balloon"])
        other = pipeline(["a red balloon"], seed=99)
        assert not torch.equal(first.video, other.video)

    def test_a_different_prompt_gives_a_different_sample(self) -> None:
        # If this fails the text tower is not reaching the model, and every
        # guidance assertion below would be vacuous.
        pipeline = build_pipeline()
        assert not torch.equal(
            pipeline(["a red balloon"]).video, pipeline(["a blue kite"]).video
        )

    def test_the_pinned_config_records_what_actually_ran(self) -> None:
        pipeline = build_pipeline(steps=4)
        media = pipeline(["a red balloon"], schedule=ScheduleConfig(steps=4, shift=3.0))
        assert media.config["resolved_shift"] == 3.0
        assert media.config["steps"] == 4
        assert media.config["video_codec_id"] == pipeline.video_codec.fingerprint
        assert media.config["text_encoder_id"] == pipeline.text_encoder.fingerprint
        assert media.config["latent_statistics"] == pipeline.latent_statistics.to_dict()
        assert media.config["sequence_length"] > 0

    def test_the_metadata_is_json_safe(self) -> None:
        import json

        media = build_pipeline()(["a red balloon"])
        # Round-tripping through json is the actual requirement: the record is
        # written next to the media file.
        assert json.loads(json.dumps(media.metadata()))["fps"] == 8.0

    def test_the_step_count_override_reaches_the_schedule(self) -> None:
        # steps lives in two places and the schedule is what builds the sigmas,
        # so `steps=8` has to do what the caller meant.
        seen: list[int] = []
        build_pipeline()(
            ["a red balloon"],
            steps=8,
            progress_callback=lambda index, total, sigma: seen.append(total),
        )
        assert seen == [8] * 8

    def test_the_callbacks_fire_once_per_step(self) -> None:
        indices: list[int] = []
        shapes: list[tuple[int, ...]] = []
        build_pipeline(steps=5)(
            ["a red balloon"],
            progress_callback=lambda index, total, sigma: indices.append(index),
            latent_callback=lambda index, tensor: shapes.append(tuple(tensor.shape)),
        )
        assert indices == [0, 1, 2, 3, 4]
        assert shapes == [(1, 4, 4, 4, 4)] * 5

    def test_returning_latents_returns_the_normalised_grid(self) -> None:
        media = build_pipeline()(["a red balloon"], return_latents=True)
        assert media.latents is not None
        assert tuple(media.latents.shape) == (1, 4, 4, 4, 4)

    @pytest.mark.parametrize("name", list_samplers())
    def test_every_sampler_drives_the_pipeline(self, name: str) -> None:
        media = build_pipeline()(
            ["a red balloon"],
            sampler=SamplerConfig(
                name=name, eta=0.5 if name == "euler_ancestral" else 0.0
            ),
        )
        assert bool(torch.isfinite(media.video).all())
        assert tuple(media.video.shape) == (1, 1, 4, 8, 8)

    @pytest.mark.parametrize("name", list_sigma_schedules())
    def test_every_schedule_drives_the_pipeline(self, name: str) -> None:
        media = build_pipeline()(
            ["a red balloon"], schedule=ScheduleConfig(name=name, steps=3, shift=2.0)
        )
        assert bool(torch.isfinite(media.video).all())

    def test_guidance_at_scale_one_is_bit_identical_to_unguided(self) -> None:
        # The property that lets a caller switch guidance off without a separate
        # code path, and without the null forward pass.
        pipeline = build_pipeline()
        unguided = pipeline(["a red balloon"], guidance=GuidanceConfig(scale=1.0))
        bare = pipeline(["a red balloon"], guidance=1.0)
        assert torch.equal(unguided.video, bare.video)

    def test_guidance_above_one_changes_the_sample(self) -> None:
        pipeline = build_pipeline()
        unguided = pipeline(["a red balloon"], guidance=GuidanceConfig(scale=1.0))
        guided = pipeline(["a red balloon"], guidance=GuidanceConfig(scale=6.0))
        assert not torch.equal(unguided.video, guided.video)

    @pytest.mark.parametrize(
        "guidance",
        [
            GuidanceConfig(scale=6.0, rescale=0.7),
            GuidanceConfig(scale=6.0, projection=True),
            GuidanceConfig(scale=6.0, projection=True, projection_threshold=2.0),
            GuidanceConfig(scale=6.0, schedule="cosine", min_scale=1.0),
            GuidanceConfig(
                scale=6.0, schedule="interval", start_fraction=0.2, end_fraction=0.8
            ),
            GuidanceConfig(scale=4.0, text_scale=1.5, audio_scale=0.5),
        ],
    )
    def test_every_guidance_variant_produces_finite_media(
        self, guidance: GuidanceConfig
    ) -> None:
        media = build_pipeline()(["a red balloon"], guidance=guidance)
        assert bool(torch.isfinite(media.video).all())

    def test_a_negative_prompt_replaces_the_zeroed_null_branch(self) -> None:
        # The model is steered *away* from the negative prompt rather than
        # merely away from nothing, so the two must not coincide.
        pipeline = build_pipeline()
        null_branch = pipeline(["a red balloon"], guidance=GuidanceConfig(scale=6.0))
        steered = pipeline(
            ["a red balloon"],
            guidance=GuidanceConfig(scale=6.0),
            negative_prompts="blurry",
        )
        assert steered.negative_prompts == ("blurry",)
        assert not torch.equal(null_branch.video, steered.video)

    def test_a_per_sample_negative_prompt_list_is_honoured(self) -> None:
        media = build_pipeline()(
            ["a red balloon", "a blue kite"],
            guidance=GuidanceConfig(scale=4.0),
            negative_prompts=["blurry", "dark"],
        )
        assert media.negative_prompts == ("blurry", "dark")

    def test_a_mismatched_negative_prompt_count_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_pipeline()(
                ["a", "b", "c"],
                guidance=GuidanceConfig(scale=4.0),
                negative_prompts=["blurry", "dark"],
            )

    def test_audio_is_generated_and_decoded_at_the_codecs_rate(self) -> None:
        media = build_pipeline()(["a red balloon"], generate_audio=True)
        assert media.has_audio
        assert media.audio is not None
        assert tuple(media.audio.shape)[:2] == (1, 1)
        assert media.sample_rate == 1600
        assert bool(torch.isfinite(media.audio).all())

    def test_audio_without_a_codec_is_refused_at_construction(self) -> None:
        # Refusing here rather than at the decode is what stops a long sampling
        # run from being thrown away at its last step.
        with pytest.raises(ValueError, match="no audio_codec was supplied"):
            GenerationPipeline(
                build_denoiser(responsive=False),
                video_codec=ReferenceVideoCodec(channels=1, spatial_compression=2),
                config=GenerationConfig(generate_audio=True),
            )

    def test_the_pipeline_takes_the_patchifier_from_the_model(self) -> None:
        # The token layout has to match the weights, so defaulting to the
        # model's own patchifier is the only safe default.
        model = build_denoiser()
        pipeline = build_pipeline(model=model)
        assert pipeline.patchifier == model.patchifier

    def test_without_a_text_encoder_generation_is_unconditional(self) -> None:
        model = build_denoiser()
        pipeline = GenerationPipeline(
            model,
            video_codec=ReferenceVideoCodec(
                channels=1, temporal_compression=1, spatial_compression=2
            ),
            config=GenerationConfig(
                steps=2,
                height=8,
                width=8,
                num_frames=4,
                schedule=ScheduleConfig(steps=2),
            ),
            device=torch.device("cpu"),
        )
        media = pipeline(["ignored"])
        assert bool(torch.isfinite(media.video).all())
        assert media.config["text_encoder_id"] is None

    def test_survives_a_config_serialisation_round_trip(self) -> None:
        config = GenerationConfig(
            steps=7,
            seed=3,
            guidance=GuidanceConfig(scale=6.0, rescale=0.7),
            sampler=SamplerConfig(name="heun"),
            schedule=ScheduleConfig(name="karras", steps=7),
        )
        assert GenerationConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"steps": 0},
            {"height": 0},
            {"num_frames": -1},
            {"fps": 0.0},
            {"seed": -1},
            {"audio_seconds": 0.0},
        ],
    )
    def test_rejects_an_invalid_generation_config(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(**kwargs)


class TestFromCheckpoint:
    """A checkpoint must carry the settings that make its seed reproducible.

    The seed alone reproduces nothing if the shift, the sampler, or the guidance
    schedule has moved since — and the shift is the field most likely to be got
    wrong, because nothing about a wrong one is visible in the sampler. Shipping
    the generation defaults next to the weights is what closes that gap, so this
    checks that they actually survive the trip.
    """

    @staticmethod
    def write_checkpoint(root: Path, *, generation: dict[str, Any] | None) -> VideoDiT:
        """Write a tiny release-format checkpoint and return the source model."""
        from safetensors.torch import save_file

        torch.manual_seed(0)
        config = preset("tiny")
        model = VideoDiT(config)
        model.init_weights()
        state = {key: value.contiguous() for key, value in model.state_dict().items()}
        save_file(state, str(root / "model.safetensors"))
        settings = {
            key: value
            for key, value in dataclasses.asdict(config).items()
            if key != "rope_scaling"
        }
        manifest: dict[str, Any] = {"model": {"name": "video_dit", "config": settings}}
        if generation is not None:
            manifest["generation"] = generation
        (root / "config.json").write_text(json.dumps(manifest))
        return model

    def test_the_weights_and_the_generation_defaults_both_load(
        self, tmp_path: Path
    ) -> None:
        pinned = GenerationConfig(
            steps=2,
            seed=5,
            height=8,
            width=8,
            num_frames=4,
            fps=8.0,
            schedule=ScheduleConfig(name="linear", steps=2, shift=3.0),
        )
        source = self.write_checkpoint(tmp_path, generation=pinned.to_dict())
        pipeline = GenerationPipeline.from_checkpoint(
            tmp_path,
            video_codec=ReferenceVideoCodec(
                channels=1, temporal_compression=1, spatial_compression=2
            ),
            text_encoder=ReferenceTextEncoder(width=32, max_length=8),
            device="cpu",
        )
        assert pipeline.config == pinned
        loaded = pipeline.model.state_dict()
        for key, value in source.state_dict().items():
            assert torch.equal(loaded[key], value), f"{key} did not load"
        media = pipeline(["a red balloon"])
        # The shift the checkpoint shipped is the shift the sample was drawn at.
        assert media.config["resolved_shift"] == 3.0

    def test_an_explicit_config_overrides_the_checkpoints_defaults(
        self, tmp_path: Path
    ) -> None:
        # The checkpoint's defaults are defaults, not a lock: a caller resampling
        # at a different step count must not have to edit the config.json.
        self.write_checkpoint(
            tmp_path,
            generation=GenerationConfig(steps=9, seed=11).to_dict(),
        )
        override = GenerationConfig(
            steps=2, seed=1, height=8, width=8, num_frames=4, fps=8.0
        )
        pipeline = GenerationPipeline.from_checkpoint(
            tmp_path,
            video_codec=ReferenceVideoCodec(
                channels=1, temporal_compression=1, spatial_compression=2
            ),
            config=override,
            device="cpu",
        )
        assert pipeline.config == override

    def test_a_checkpoint_without_generation_defaults_uses_the_library_ones(
        self, tmp_path: Path
    ) -> None:
        self.write_checkpoint(tmp_path, generation=None)
        pipeline = GenerationPipeline.from_checkpoint(
            tmp_path,
            video_codec=ReferenceVideoCodec(
                channels=1, temporal_compression=1, spatial_compression=2
            ),
            device="cpu",
        )
        assert pipeline.config == GenerationConfig()

    def test_a_partial_export_is_refused_rather_than_half_loaded(
        self, tmp_path: Path
    ) -> None:
        # Loading non-strictly is deliberate, but silently leaving half a model
        # randomly initialised produces samples that look like a bad checkpoint
        # rather than like a loading bug.
        from safetensors.torch import save_file

        self.write_checkpoint(tmp_path, generation=None)
        state = VideoDiT(preset("tiny")).state_dict()
        first = next(iter(state))
        save_file(
            {first: state[first].contiguous()},
            str(tmp_path / "model.safetensors"),
        )
        with pytest.raises(RuntimeError, match="parameters were not found"):
            GenerationPipeline.from_checkpoint(
                tmp_path,
                video_codec=ReferenceVideoCodec(channels=1, spatial_compression=2),
            )

    def test_a_missing_path_is_reported_before_anything_is_built(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            GenerationPipeline.from_checkpoint(
                tmp_path / "absent",
                video_codec=ReferenceVideoCodec(channels=1, spatial_compression=2),
            )

    def test_a_checkpoint_without_a_model_name_says_so(self, tmp_path: Path) -> None:
        # A precise message rather than an untrained model: the alternative
        # failure mode is a pipeline that runs and produces noise.
        (tmp_path / "model.safetensors").touch()
        with pytest.raises(RuntimeError, match="no model name found"):
            GenerationPipeline.from_checkpoint(
                tmp_path,
                video_codec=ReferenceVideoCodec(channels=1, spatial_compression=2),
            )


@pytest.mark.gpu
@CUDA
class TestOnDevice:
    """Device placement and device-RNG determinism.

    Everything else in this file is device-independent arithmetic. These check
    the two things a CPU run cannot: that the schedule, the sampler noise, and
    the generated media land on the requested device, and that the seed contract
    holds against the CUDA generator rather than only the CPU one.
    """

    def test_a_schedule_builds_and_moves_between_devices(self) -> None:
        schedule = build_sigma_schedule(
            ScheduleConfig(name="karras", steps=6), device="cuda"
        )
        assert schedule.device.type == "cuda"
        assert schedule.to("cpu").device.type == "cpu"
        assert float(schedule.sigmas[-1]) == 0.0

    def test_the_constant_field_integrates_exactly_on_cuda(self) -> None:
        torch.manual_seed(0)
        schedule = build_sigma_schedule(ScheduleConfig(name="linear", steps=12))
        clean = torch.randn(2, 3, 4, device="cuda")
        noise = torch.randn(2, 3, 4, device="cuda")
        start_sigma = float(schedule.sigmas[0])
        start = (1.0 - start_sigma) * clean + start_sigma * noise
        result = integrate("dpmpp_2m", constant_field(noise - clean), start, schedule)
        assert result.device.type == "cuda"
        error = float((result - clean).abs().max())
        assert error < SOLVER_EXACT, f"cuda solver drifted by {error:.3e}"

    def test_the_ancestral_seed_contract_holds_against_the_cuda_generator(self) -> None:
        solver = EulerAncestralSampler(SamplerConfig(name="euler_ancestral", eta=0.7))
        x_t = torch.randn(2, 5, device="cuda")
        velocity = torch.randn(2, 5, device="cuda")

        def run(seed: int) -> torch.Tensor:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            return solver.step(velocity, x_t, 0.8, 0.5, generator=generator)

        assert torch.equal(run(5), run(5))
        assert not torch.equal(run(5), run(6))

    def test_the_pipeline_generates_on_cuda(self) -> None:
        model = build_denoiser().to("cuda")
        pipeline = GenerationPipeline(
            model,
            video_codec=ReferenceVideoCodec(
                channels=1, temporal_compression=1, spatial_compression=2
            ),
            text_encoder=ReferenceTextEncoder(width=32, max_length=8, device="cuda"),
            config=GenerationConfig(
                steps=2,
                seed=1,
                height=8,
                width=8,
                num_frames=4,
                fps=8.0,
                schedule=ScheduleConfig(name="linear", steps=2),
            ),
            device=torch.device("cuda"),
        )
        media = pipeline(["a red balloon"])
        assert media.video.device.type == "cuda"
        assert bool(torch.isfinite(media.video).all())
        assert torch.equal(pipeline(["a red balloon"]).video, media.video)
