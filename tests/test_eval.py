"""Tests for the evaluation subsystem.

Two properties dominate this file, and both are about evidence rather than
arithmetic.

**A metric that runs is not a metric that measures.** Every dependency-free
metric here has a value that theory pins down — flicker index near 3 on
temporally independent frames, temporal consistency exactly 1 on a frozen clip,
Fréchet distance exactly 0 between a set and itself — and those are the
assertions that catch a metric which executes happily and measures nothing.

**A report that does not carry its sampling settings is not evidence.** The
pin, and ``compare_reports`` refusing to compare across it, is the mechanism
that makes an invalid comparison inconvenient. Most published
video-generation comparisons are invalid for exactly the reason these tests
enumerate, so every pinned field gets its own assertion.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from avgen.eval import (
    EvalBatch,
    EvalPin,
    EvalReport,
    IncomparableReports,
    LearnedMetricSpec,
    MetricBundle,
    MetricError,
    MissingBackendError,
    RunningMetric,
    available_learned_metrics,
    build_learned_metric,
    build_metric,
    build_metrics,
    capture_environment,
    compare_reports,
    describe_learned_metrics,
    describe_metrics,
    format_comparison,
    frechet_distance,
    learned_metric_available,
    learned_metric_summary,
    list_learned_metrics,
    list_metrics,
    metric_class,
    prompt_set_digest,
    register_learned_metric,
    register_metric,
    require_backend,
    run_eval_suite,
    shard_prompts,
)
from avgen.eval.learned import merge_feature_banks

#: Statistic keys each built-in must report when handed only its required
#: inputs. Written out rather than derived, because "the documented keys" is
#: exactly the thing a refactor can change without anyone noticing — a renamed
#: statistic silently breaks every plot and every saved report.
EXPECTED_KEYS: dict[str, tuple[str, ...]] = {
    "audio_bandwidth": ("centroid", "rolloff"),
    "audio_silence": ("frame_rate", "clip_rate", "clipped_fraction"),
    "av_sync_proxy": ("peak_correlation", "zero_lag_correlation", "lag_frames"),
    "first_frame_fidelity": ("mse", "psnr_db", "cosine"),
    "flicker_index": ("index", "second_difference"),
    "inpaint_boundary": ("gradient_ratio",),
    "motion_magnitude": ("mean", "peak"),
    "saturation": ("clipped_fraction", "out_of_range_fraction"),
    "seam_continuity": ("ratio", "seam_delta"),
    "sharpness": ("mean", "decay"),
    "static_frames": ("pair_rate", "clip_rate"),
    "temporal_consistency": ("cosine", "worst_pair"),
}

BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH = 2, 3, 12, 8, 8


def random_video(*, seed: int = 0, frames: int = FRAMES) -> torch.Tensor:
    """Reproducible i.i.d. video, the worst case every metric must survive."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, CHANNELS, frames, HEIGHT, WIDTH, generator=generator)


def static_video(*, batch: int = BATCH, frames: int = FRAMES) -> torch.Tensor:
    """A clip that is spatially textured and perfectly frozen in time."""
    generator = torch.Generator().manual_seed(7)
    frame = torch.randn(batch, CHANNELS, 1, HEIGHT, WIDTH, generator=generator)
    return frame.expand(batch, CHANNELS, frames, HEIGHT, WIDTH).contiguous()


def inputs_for(metric: RunningMetric, *, seed: int = 0) -> dict[str, Any]:
    """Build exactly the inputs a metric declares it requires."""
    generator = torch.Generator().manual_seed(seed + 1000)
    supplied: dict[str, Any] = {}
    for key in metric.required_inputs:
        if key == "video":
            supplied["video"] = random_video(seed=seed)
        elif key == "audio":
            supplied["audio"] = torch.randn(BATCH, 2, 64, generator=generator)
        elif key == "reference":
            supplied["reference"] = random_video(seed=seed + 500)
        elif key == "mask":
            mask = torch.zeros(BATCH, 1, FRAMES, HEIGHT, WIDTH)
            mask[:, :, :, 2:6, 2:6] = 1.0
            supplied["mask"] = mask
        else:  # pragma: no cover - a new required input needs a new branch
            raise AssertionError(f"no fixture for required input {key!r}")
    return supplied


class TestMetricRegistry:
    """The registry is the contract between a YAML file and a number.

    A metric name in ``eval.metrics`` has to resolve to exactly one
    implementation, forever. Two implementations behind one name means two runs
    report incomparable numbers under the same label, which is worse than a
    crash.
    """

    def test_the_registry_is_populated(self) -> None:
        assert len(list_metrics()) >= 10
        assert list_metrics() == tuple(sorted(list_metrics()))

    def test_the_expected_key_table_covers_every_registered_metric(self) -> None:
        # If this fails, a metric was added or renamed and the table below has
        # not been updated — which means the new metric is untested.
        assert set(EXPECTED_KEYS) == set(list_metrics())

    def test_metric_class_lookup(self) -> None:
        assert metric_class("flicker_index").name == "flicker_index"

    def test_an_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(MetricError, match=r"unknown metric 'nope'"):
            metric_class("nope")
        with pytest.raises(MetricError, match=r"avgen.eval.learned"):
            build_metric("nope")

    def test_bad_constructor_arguments_name_the_metric(self) -> None:
        with pytest.raises(MetricError, match=r"cannot build metric"):
            build_metric("flicker_index", not_a_real_argument=1)

    def test_registering_a_duplicate_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"already registered"):

            @register_metric("flicker_index")
            class _Shadow(RunningMetric):
                def observe(self, **inputs: Any) -> dict[str, Any]:
                    return {}

    def test_describe_metrics_returns_one_line_each(self) -> None:
        described = describe_metrics()
        assert set(described) == set(list_metrics())
        assert all(line and "\n" not in line for line in described.values())


class TestDependencyFreeMetrics:
    """Every built-in must be finite, resettable, and stable on random input.

    i.i.d. noise is the adversarial case for these: it maximises motion, has no
    temporal structure, and drives every normalisation toward its guard. A NaN
    here poisons the whole report, and a report with one NaN in it gets thrown
    away entirely.
    """

    @pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
    def test_produces_the_documented_keys(self, name: str) -> None:
        metric = build_metric(name)
        metric.update(**inputs_for(metric))
        expected = {f"{name}/{key}" for key in EXPECTED_KEYS[name]}
        assert set(metric.compute()) == expected

    @pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
    def test_every_value_is_finite(self, name: str) -> None:
        metric = build_metric(name)
        metric.update(**inputs_for(metric))
        values = metric.compute()
        assert values
        for key, value in values.items():
            assert math.isfinite(value), f"{key} is {value}"

    @pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
    def test_compute_is_idempotent(self, name: str) -> None:
        # A report is rendered and then written; the second read must agree.
        metric = build_metric(name)
        metric.update(**inputs_for(metric))
        assert metric.compute() == metric.compute()

    @pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
    def test_reset_clears_every_accumulator(self, name: str) -> None:
        metric = build_metric(name)
        metric.update(**inputs_for(metric))
        assert metric.compute()
        metric.reset()
        assert metric.compute() == {}
        assert metric.state() == {}

    @pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
    def test_a_missing_required_input_names_the_metric(self, name: str) -> None:
        metric = build_metric(name)
        with pytest.raises(MetricError, match=name):
            metric.update()

    @pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
    def test_a_fresh_metric_computes_nothing(self, name: str) -> None:
        # An empty mapping is how the suite tells "skipped" from "measured 0".
        assert build_metric(name).compute() == {}


class TestSanityValues:
    """Values theory pins down. These catch a metric that measures nothing.

    A metric can execute, return a finite float, and be completely disconnected
    from the property its name claims. The only defence is a case where the
    right answer is known in advance.
    """

    def test_temporal_consistency_is_one_on_a_frozen_clip(self) -> None:
        # And that is exactly why it must never be reported alone: a still
        # image wins the metric that looks like it guards against still images.
        metric = build_metric("temporal_consistency")
        metric.update(video=static_video())
        assert metric.compute()["temporal_consistency/cosine"] == pytest.approx(
            1.0, abs=1e-5
        )

    def test_temporal_consistency_is_near_zero_on_independent_frames(self) -> None:
        metric = build_metric("temporal_consistency")
        metric.update(video=random_video(frames=32))
        assert abs(metric.compute()["temporal_consistency/cosine"]) < 0.1

    def test_motion_magnitude_is_zero_on_a_frozen_clip(self) -> None:
        metric = build_metric("motion_magnitude")
        metric.update(video=static_video())
        assert metric.compute()["motion_magnitude/mean"] == pytest.approx(0.0, abs=1e-6)

    def test_motion_magnitude_is_large_on_independent_frames(self) -> None:
        metric = build_metric("motion_magnitude")
        metric.update(video=random_video(frames=32))
        assert metric.compute()["motion_magnitude/mean"] > 0.5

    def test_flicker_index_is_three_on_temporally_independent_frames(self) -> None:
        # For i.i.d. frames of variance s the first difference has variance 2s
        # and the second 6s, so the ratio is 3. A reading near 3 means the model
        # learned no temporal structure at all — a different diagnosis from
        # "the motion is wrong", and this is the number that distinguishes them.
        metric = build_metric("flicker_index")
        metric.update(
            video=torch.randn(
                8, 3, 64, 8, 8, generator=torch.Generator().manual_seed(3)
            )
        )
        assert metric.compute()["flicker_index/index"] == pytest.approx(3.0, rel=0.06)

    def test_flicker_index_is_zero_under_constant_velocity(self) -> None:
        ramp = torch.arange(24.0).view(1, 1, 24, 1, 1).expand(2, 3, 24, 8, 8)
        metric = build_metric("flicker_index")
        metric.update(video=ramp.contiguous())
        assert metric.compute()["flicker_index/index"] == pytest.approx(0.0, abs=1e-6)

    def test_static_frame_ratio_separates_frozen_from_moving(self) -> None:
        frozen = build_metric("static_frames")
        frozen.update(video=static_video())
        assert frozen.compute()["static_frames/clip_rate"] == 1.0
        assert frozen.compute()["static_frames/pair_rate"] == 1.0

        moving = build_metric("static_frames")
        moving.update(video=random_video(frames=32))
        assert moving.compute()["static_frames/clip_rate"] == 0.0
        assert moving.compute()["static_frames/pair_rate"] == 0.0

    def test_saturation_is_zero_inside_the_range_and_one_when_pinned(self) -> None:
        clean = build_metric("saturation")
        clean.update(video=torch.zeros(BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH))
        assert clean.compute()["saturation/clipped_fraction"] == 0.0

        blown = build_metric("saturation")
        blown.update(video=torch.ones(BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH))
        assert blown.compute()["saturation/clipped_fraction"] == 1.0
        assert blown.compute()["saturation/out_of_range_fraction"] == 0.0

        outside = build_metric("saturation")
        outside.update(video=torch.full((BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH), 2.0))
        assert outside.compute()["saturation/out_of_range_fraction"] == 1.0

    def test_first_frame_fidelity_is_perfect_against_itself(self) -> None:
        # I2V fidelity is near-binary in practice: the conditioning path works
        # or it does not. The identity case is the "works" end of that scale.
        video = random_video()
        metric = build_metric("first_frame_fidelity")
        metric.update(video=video, reference=video.clone())
        values = metric.compute()
        assert values["first_frame_fidelity/mse"] == pytest.approx(0.0, abs=1e-10)
        assert values["first_frame_fidelity/cosine"] == pytest.approx(1.0, abs=1e-5)
        assert values["first_frame_fidelity/psnr_db"] > 80.0

    def test_seam_continuity_is_about_one_without_a_seam(self) -> None:
        metric = build_metric("seam_continuity", context_frames=4)
        metric.update(video=random_video(frames=16))
        assert metric.compute()["seam_continuity/ratio"] == pytest.approx(1.0, rel=0.2)

    def test_seam_continuity_flags_a_hard_cut(self) -> None:
        clip = static_video(frames=16).clone()
        clip[:, :, 8:] = -clip[:, :, 8:]  # a hard cut exactly at the seam
        metric = build_metric("seam_continuity", context_frames=8)
        metric.update(video=clip)
        assert metric.compute()["seam_continuity/ratio"] > 4.0

    def test_sharpness_decay_is_one_on_a_uniform_clip(self) -> None:
        metric = build_metric("sharpness")
        metric.update(video=random_video(frames=24))
        assert metric.compute()["sharpness/decay"] == pytest.approx(1.0, rel=0.15)

    def test_sharpness_decay_falls_when_detail_bleeds_away(self) -> None:
        # The signature failure of continuation and long-video-by-extension
        # models: each chunk conditions on a blurrier one. Invisible in FVD.
        clip = random_video(frames=24).clone()
        clip[:, :, 16:] *= 0.05  # last third has almost no high-frequency energy
        metric = build_metric("sharpness")
        metric.update(video=clip)
        assert metric.compute()["sharpness/decay"] < 0.5

    def test_audio_silence_detects_a_silent_clip(self) -> None:
        # A model that emits silence scores well on many published sync
        # metrics, because a flat signal correlates with nothing.
        metric = build_metric("audio_silence")
        metric.update(audio=torch.zeros(BATCH, 2, 64))
        values = metric.compute()
        assert values["audio_silence/clip_rate"] == 1.0
        assert values["audio_silence/clipped_fraction"] == 0.0

    def test_audio_silence_is_zero_on_live_audio(self) -> None:
        metric = build_metric("audio_silence")
        metric.update(
            audio=torch.randn(BATCH, 2, 256, generator=torch.Generator().manual_seed(1))
        )
        assert metric.compute()["audio_silence/clip_rate"] == 0.0

    def test_av_sync_proxy_finds_a_planted_lag(self) -> None:
        # The honest use of this metric: a peak at a non-zero lag means the
        # model learned the association and put it in the wrong place.
        frames = 24
        pulse = torch.zeros(1, 1, frames, HEIGHT, WIDTH)
        for index in range(0, frames, 4):
            pulse[:, :, index] = 1.0
        video = pulse.expand(1, CHANNELS, frames, HEIGHT, WIDTH).contiguous()
        # Motion energy lives on frame pairs, so build the audio envelope on the
        # same grid and shift it by two frames.
        envelope = torch.zeros(1, 1, frames - 1)
        for index in range(0, frames - 1, 4):
            envelope[:, :, index] = 1.0
        shifted = torch.roll(envelope, shifts=2, dims=-1)
        metric = build_metric("av_sync_proxy", max_lag=4)
        metric.update(video=video, audio=shifted)
        values = metric.compute()
        assert values["av_sync_proxy/peak_correlation"] > 0.5
        assert values["av_sync_proxy/lag_frames"] == pytest.approx(2.0)

    def test_inpaint_boundary_reports_leakage_only_with_a_reference(self) -> None:
        video = random_video()
        mask = torch.zeros(BATCH, 1, FRAMES, HEIGHT, WIDTH)
        mask[:, :, :, 2:6, 2:6] = 1.0

        without = build_metric("inpaint_boundary")
        without.update(video=video, mask=mask)
        assert "inpaint_boundary/leakage" not in without.compute()

        with_reference = build_metric("inpaint_boundary")
        with_reference.update(video=video, mask=mask, reference=video.clone())
        values = with_reference.compute()
        # Identical outside the mask means no leakage, by construction.
        assert values["inpaint_boundary/leakage"] == pytest.approx(0.0, abs=1e-9)


class TestMetricInputValidation:
    """A wrong tensor rank must be refused, not silently reinterpreted."""

    def test_a_non_tensor_video(self) -> None:
        metric = build_metric("motion_magnitude")
        with pytest.raises(MetricError, match=r"must be a tensor"):
            metric.update(video=[1, 2, 3])

    def test_a_four_dimensional_video(self) -> None:
        metric = build_metric("motion_magnitude")
        with pytest.raises(MetricError, match=r"batch, channels, frames"):
            metric.update(video=torch.randn(2, 3, 8, 8))

    def test_a_two_dimensional_audio(self) -> None:
        metric = build_metric("audio_silence")
        with pytest.raises(MetricError, match=r"batch, channels, frames"):
            metric.update(audio=torch.randn(2, 8))

    def test_a_reference_of_a_different_shape(self) -> None:
        metric = build_metric("first_frame_fidelity")
        with pytest.raises(MetricError, match=r"same shape"):
            metric.update(video=random_video(), reference=random_video(frames=8))

    def test_a_frame_index_past_the_end(self) -> None:
        metric = build_metric("first_frame_fidelity", frame_index=99)
        with pytest.raises(MetricError, match=r"out of range"):
            metric.update(video=random_video(), reference=random_video())

    def test_a_mask_that_does_not_align(self) -> None:
        metric = build_metric("inpaint_boundary")
        with pytest.raises(MetricError, match=r"does not align"):
            metric.update(video=random_video(), mask=torch.ones(BATCH, 1, FRAMES, 4, 4))

    def test_a_mask_that_is_not_a_tensor(self) -> None:
        metric = build_metric("inpaint_boundary")
        with pytest.raises(MetricError, match=r"mask must be"):
            metric.update(video=random_video(), mask=[[0, 1]])

    def test_a_mask_of_none_is_reported_as_an_absent_input(self) -> None:
        # `None` means "this evaluation did not produce one", which is a
        # different message from "the one you gave me is the wrong shape".
        metric = build_metric("inpaint_boundary")
        with pytest.raises(MetricError, match=r"requires mask"):
            metric.update(video=random_video(), mask=None)

    def test_a_context_that_leaves_no_continuation(self) -> None:
        metric = build_metric("seam_continuity")
        with pytest.raises(MetricError, match=r"leaves no continuation"):
            metric.update(video=random_video(), context_frames=FRAMES)

    @pytest.mark.parametrize(
        ("name", "kwargs", "match"),
        [
            ("temporal_consistency", {"stride": 0}, "stride must be >= 1"),
            ("static_frames", {"threshold": 0.0}, "threshold must be > 0"),
            ("saturation", {"value_range": (1.0, 1.0)}, "must be increasing"),
            ("seam_continuity", {"context_frames": 0}, "context_frames must be >= 1"),
            ("first_frame_fidelity", {"frame_index": -1}, "frame_index must be >= 0"),
            ("av_sync_proxy", {"max_lag": 0}, "max_lag must be >= 1"),
            ("av_sync_proxy", {"video_fps": -1.0}, "video_fps must be >= 0"),
            ("audio_silence", {"silence_threshold": 1.5}, r"must be in \(0, 1\)"),
            ("audio_silence", {"clip_threshold": 0.0}, "clip_threshold must be > 0"),
            ("audio_bandwidth", {"rolloff_fraction": 0.0}, r"must be in \(0, 1\)"),
            ("audio_bandwidth", {"sample_rate": -1.0}, "sample_rate must be >= 0"),
        ],
    )
    def test_constructor_arguments_are_validated(
        self, name: str, kwargs: dict[str, Any], match: str
    ) -> None:
        with pytest.raises((ValueError, MetricError), match=match):
            build_metric(name, **kwargs)


class TestShortClips:
    """Too-short input produces no statistic, never a fabricated one.

    Returning zero for a clip with one frame would be indistinguishable from a
    model that produced a perfectly still video, which is the exact failure the
    metric exists to detect.
    """

    @pytest.mark.parametrize(
        "name", ["motion_magnitude", "static_frames", "temporal_consistency"]
    )
    def test_a_single_frame_yields_nothing(self, name: str) -> None:
        metric = build_metric(name)
        metric.update(video=torch.randn(1, 3, 1, 8, 8))
        assert metric.compute() == {}

    def test_flicker_needs_three_frames(self) -> None:
        metric = build_metric("flicker_index")
        metric.update(video=torch.randn(1, 3, 2, 8, 8))
        assert metric.compute() == {}

    def test_sharpness_needs_a_three_pixel_axis(self) -> None:
        metric = build_metric("sharpness")
        metric.update(video=torch.randn(1, 3, 8, 2, 2))
        assert metric.compute() == {}

    def test_av_sync_needs_three_frames_of_each(self) -> None:
        metric = build_metric("av_sync_proxy")
        metric.update(video=torch.randn(1, 3, 2, 8, 8), audio=torch.randn(1, 2, 2))
        assert metric.compute() == {}

    def test_a_mask_covering_everything_has_no_boundary(self) -> None:
        metric = build_metric("inpaint_boundary")
        metric.update(
            video=random_video(), mask=torch.ones(BATCH, 1, FRAMES, HEIGHT, WIDTH)
        )
        assert metric.compute() == {}


class TestAccumulationContract:
    """Statistics are averaged over samples, never over batches.

    Averaging per-batch means over-weights a short final batch. On the 50-prompt
    sets people actually use that is several percent — the same size as the
    differences being reported.
    """

    def test_the_mean_is_sample_weighted_not_batch_weighted(self) -> None:
        metric = build_metric("static_frames")
        metric.update(video=static_video(batch=4))  # four frozen clips -> 1.0
        metric.update(video=random_video(frames=FRAMES)[:1])  # one moving -> 0.0
        pair_rate = metric.compute()["static_frames/pair_rate"]
        assert pair_rate == pytest.approx(4.0 / 5.0)
        assert pair_rate != pytest.approx(0.5)  # the batch-mean answer

    def test_state_exposes_raw_sums_and_counts(self) -> None:
        metric = build_metric("static_frames")
        metric.update(video=static_video(batch=4))
        total, count = metric.state()["clip_rate"]
        assert (total, count) == (4.0, 4.0)

    def test_merge_reproduces_a_single_rank_result(self) -> None:
        # A distributed suite sums sufficient statistics; averaging per-rank
        # compute() outputs is wrong the moment ranks see different counts.
        combined = build_metric("static_frames")
        combined.update(video=static_video(batch=4))
        combined.update(video=random_video()[:1])

        rank_a = build_metric("static_frames")
        rank_a.update(video=static_video(batch=4))
        rank_b = build_metric("static_frames")
        rank_b.update(video=random_video()[:1])
        rank_a.merge(rank_b.state())

        assert rank_a.compute() == pytest.approx(combined.compute())

    def test_merging_into_an_empty_metric_works(self) -> None:
        source = build_metric("static_frames")
        source.update(video=static_video(batch=2))
        target = build_metric("static_frames")
        target.merge(source.state())
        assert target.compute() == source.compute()

    def test_accumulation_is_float64(self) -> None:
        # An evaluation loop accumulates tens of thousands of terms; fp32
        # summation visibly drifts over that many additions.
        metric = build_metric("motion_magnitude")
        metric.update(video=random_video())
        assert all(tensor.dtype is torch.float64 for tensor in metric._sums.values())


class TestFrechetDistance:
    """The shared core of FID, FVD, and every other "Fréchet X distance"."""

    def test_a_set_against_itself_is_zero(self) -> None:
        features = torch.randn(128, 16, generator=torch.Generator().manual_seed(0))
        assert frechet_distance(features, features) == pytest.approx(0.0, abs=1e-6)

    def test_it_is_symmetric(self) -> None:
        generator = torch.Generator().manual_seed(1)
        first = torch.randn(96, 12, generator=generator)
        second = torch.randn(96, 12, generator=generator) * 2.0 + 1.0
        forward = frechet_distance(first, second)
        backward = frechet_distance(second, first)
        assert forward == pytest.approx(backward, rel=1e-6)

    def test_a_mean_shift_shows_up_as_its_squared_norm(self) -> None:
        generator = torch.Generator().manual_seed(2)
        features = torch.randn(4096, 8, generator=generator)
        shifted = features + 3.0
        # Same covariance, mean apart by 3 in each of 8 dimensions.
        assert frechet_distance(features, shifted) == pytest.approx(
            8 * 3.0**2, rel=0.05
        )

    def test_it_grows_with_distributional_distance(self) -> None:
        generator = torch.Generator().manual_seed(3)
        base = torch.randn(256, 16, generator=generator)
        near = torch.randn(256, 16, generator=generator) + 0.5
        far = torch.randn(256, 16, generator=generator) + 5.0
        assert frechet_distance(base, near) < frechet_distance(base, far)

    def test_mismatched_feature_dimensions_are_refused(self) -> None:
        with pytest.raises(MetricError, match=r"feature dimensions differ"):
            frechet_distance(torch.randn(8, 4), torch.randn(8, 5))

    def test_a_non_matrix_is_refused(self) -> None:
        with pytest.raises(MetricError, match=r"expects \(n, d\) matrices"):
            frechet_distance(torch.randn(8), torch.randn(8))

    def test_a_single_sample_cannot_estimate_a_covariance(self) -> None:
        with pytest.raises(MetricError, match=r"at least two samples"):
            frechet_distance(torch.randn(1, 4), torch.randn(8, 4))

    def test_merge_feature_banks_is_order_independent(self) -> None:
        banks = {"rank1": torch.ones(2, 4), "rank0": torch.zeros(3, 4)}
        merged = merge_feature_banks(banks)
        assert merged.shape == (5, 4)
        # Sorted by rank label, so the intermediate artifact is reproducible.
        assert torch.equal(merged[:3], torch.zeros(3, 4))

    def test_merge_feature_banks_rejects_mismatched_dimensions(self) -> None:
        with pytest.raises(MetricError, match=r"disagree in dimension"):
            merge_feature_banks({"a": torch.zeros(2, 4), "b": torch.zeros(2, 5)})

    def test_merge_feature_banks_rejects_nothing_at_all(self) -> None:
        with pytest.raises(MetricError, match=r"no feature banks"):
            merge_feature_banks({})


class TestLearnedMetricGating:
    """A gated metric is refused, never substituted.

    A comparison table where one column was quietly computed a different way is
    worse than a table with a hole in it, because the hole is visible.
    """

    def test_every_declared_metric_is_listed(self) -> None:
        assert set(list_learned_metrics()) >= {"fvd", "fid", "clip_score"}
        assert list_learned_metrics() == tuple(sorted(list_learned_metrics()))

    def test_an_undeclared_name_is_refused(self) -> None:
        with pytest.raises(MetricError, match=r"not a declared learned metric"):
            require_backend("not_a_metric")
        with pytest.raises(MetricError, match=r"not a declared learned metric"):
            learned_metric_available("not_a_metric")

    @pytest.mark.skipif(
        importlib.util.find_spec("transformers") is not None,
        reason="the transformers backend is installed, so it is not gated here",
    )
    def test_a_missing_backend_names_the_extra_and_the_command(self) -> None:
        with pytest.raises(MissingBackendError) as caught:
            require_backend("clip_score")
        message = str(caught.value)
        assert "transformers" in message
        assert "pip install 'avgen[text]'" in message
        # The refusal states the policy, so the reader knows it is deliberate.
        assert "will not substitute" in message

    @pytest.mark.skipif(
        importlib.util.find_spec("transformers") is not None,
        reason="the transformers backend is installed, so it is not gated here",
    )
    def test_building_a_gated_metric_never_returns_a_substitute(self) -> None:
        with pytest.raises(MissingBackendError):
            build_learned_metric("clip_score")

    def test_a_declared_metric_with_no_implementation_says_so(self) -> None:
        # fvd's declared module is torch, which is always present, so this
        # exercises the second half of the gate: backend present, no factory.
        assert learned_metric_available("fvd") is True
        with pytest.raises(MissingBackendError, match=r"ships no implementation"):
            build_learned_metric("fvd")

    def test_availability_requires_both_backend_and_implementation(self) -> None:
        # Reporting backend presence alone would call fvd "available" on any
        # machine with torch, which is true of the dependency and false of the
        # metric.
        assert available_learned_metrics()["fvd"] is False
        assert set(available_learned_metrics()) == set(list_learned_metrics())

    def test_describe_reports_a_status_and_an_install_hint(self) -> None:
        described = describe_learned_metrics()
        assert set(described) == set(list_learned_metrics())
        for detail in described.values():
            assert detail["measures"]
            assert detail["caveat"]
            assert detail["requires"]
            assert detail["status"]
        assert "bring your own" in described["fvd"]["status"]

    def test_the_fvd_caveat_states_why_numbers_do_not_transfer(self) -> None:
        caveat = describe_learned_metrics()["fvd"]["caveat"]
        assert "not comparable across implementations" in caveat

    def test_the_summary_renders_one_block_per_metric(self) -> None:
        summary = learned_metric_summary()
        for name in list_learned_metrics():
            assert name in summary
        assert "requires:" in summary

    def test_install_hint_without_an_avgen_extra(self) -> None:
        spec = LearnedMetricSpec(
            name="_probe",
            extra="",
            modules=("some_package",),
            weights="nothing",
            measures="nothing",
            caveat="nothing",
        )
        assert spec.install_hint().startswith("pip install some_package")
        assert spec.missing_modules() == ("some_package",)

    def test_registering_a_duplicate_declaration_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"already registered"):
            register_learned_metric(
                LearnedMetricSpec(
                    name="fvd",
                    extra="",
                    modules=(),
                    weights="",
                    measures="",
                    caveat="",
                )
            )


class TestEvalPin:
    """The pin is the whole mechanism. Every field in it is load-bearing."""

    def test_steps_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match=r"steps must be >= 1"):
            EvalPin(steps=0)

    def test_num_samples_may_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match=r"num_samples must be >= 0"):
            EvalPin(num_samples=-1)

    def test_to_dict_sorts_the_metric_parameters(self) -> None:
        pin = EvalPin(metrics={"b": "1", "a": "2"})
        assert list(pin.to_dict()["metrics"]) == ["a", "b"]

    def test_describe_states_the_settings_for_a_table_caption(self) -> None:
        described = EvalPin(
            sampler="heun", steps=40, guidance=7.5, seed=3, frames=16, decoded=True
        ).describe()
        assert "heun x40" in described
        assert "cfg=7.5" in described
        assert "seed=3" in described
        assert "pixels" in described
        assert "latents" in EvalPin().describe()

    def test_the_prompt_digest_is_content_addressed_and_order_sensitive(self) -> None:
        # Two runs over the same prompts in a different order draw different
        # noise per prompt at a fixed seed, so they are not the same evaluation.
        assert prompt_set_digest(["a", "b"]) == prompt_set_digest(["a", "b"])
        assert prompt_set_digest(["a", "b"]) != prompt_set_digest(["b", "a"])
        assert prompt_set_digest(["a", "b"]) != prompt_set_digest(["a", "b", "c"])
        assert prompt_set_digest(["a", "b"]).startswith("2:")
        assert prompt_set_digest([]) == "0:" + prompt_set_digest([]).split(":")[1]

    def test_a_prompt_boundary_cannot_be_forged(self) -> None:
        # The null separator is what stops ["ab"] and ["a", "b"] colliding.
        assert prompt_set_digest(["ab"]) != prompt_set_digest(["a", "b"])


#: Every pinned field, with a value that differs from the default, plus whether
#: a difference in it must invalidate a comparison.
PIN_FIELDS: list[tuple[str, Any, bool]] = [
    ("sampler", "heun", True),
    ("steps", 50, True),
    ("guidance", 7.5, True),
    ("guidance_rescale", 0.7, True),
    ("negative_prompt", "blurry", True),
    ("seed", 1, True),
    ("num_samples", 99, True),
    ("prompt_digest", "50:deadbeefdeadbeef", True),
    ("frames", 32, True),
    ("height", 64, True),
    ("width", 64, True),
    ("decoded", True, True),
    ("codec", "vae-v2", True),
    ("precision", "float32", True),
    ("metrics", {"flicker_index": "threshold=0.5"}, True),
    ("checkpoint", "step-99999", False),
]


class TestCompareReports:
    """``compare_reports`` must refuse a comparison that is not a comparison.

    A metric computed at 30 sampling steps and one computed at 50 are different
    numbers. The published video-generation literature is full of tables that
    compare them anyway; here the invalid comparison raises.
    """

    @staticmethod
    def _pair(field: str, value: Any) -> tuple[EvalReport, EvalReport]:
        base = EvalPin(
            checkpoint="step-1000",
            steps=30,
            guidance=5.0,
            seed=0,
            num_samples=64,
            prompt_digest=prompt_set_digest(["a", "b"]),
            frames=16,
            height=32,
            width=32,
        )
        left = EvalReport(pin=base, metrics={"m/x": 1.0})
        right = EvalReport(pin=replace(base, **{field: value}), metrics={"m/x": 2.0})
        return left, right

    @pytest.mark.parametrize(
        ("field", "value"),
        [(name, value) for name, value, enforced in PIN_FIELDS if enforced],
    )
    def test_a_differing_pin_field_refuses_the_comparison(
        self, field: str, value: Any
    ) -> None:
        left, right = self._pair(field, value)
        with pytest.raises(IncomparableReports) as caught:
            compare_reports(left, right)
        assert field in caught.value.differences
        assert field in str(caught.value)
        # The message has to say what to do, not merely that it refused.
        assert "allow=" in str(caught.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [(name, value) for name, value, enforced in PIN_FIELDS if enforced],
    )
    def test_the_allow_escape_hatch_permits_one_named_field(
        self, field: str, value: Any
    ) -> None:
        left, right = self._pair(field, value)
        comparison = compare_reports(left, right, allow=(field,))
        assert comparison["m/x"]["delta"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [(name, value) for name, value, enforced in PIN_FIELDS if not enforced],
    )
    def test_an_exempt_field_never_blocks_a_comparison(
        self, field: str, value: Any
    ) -> None:
        # Comparing two checkpoints is the entire point of the exercise.
        left, right = self._pair(field, value)
        assert compare_reports(left, right)

    def test_allow_does_not_relax_the_other_fields(self) -> None:
        base = EvalPin(steps=30, seed=0)
        left = EvalReport(pin=base)
        right = EvalReport(pin=replace(base, steps=50, seed=1))
        with pytest.raises(IncomparableReports) as caught:
            compare_reports(left, right, allow=("seed",))
        assert set(caught.value.differences) == {"steps"}

    def test_identical_pins_compare_cleanly(self) -> None:
        pin = EvalPin(steps=30)
        left = EvalReport(pin=pin, metrics={"a": 2.0, "b": 1.0})
        right = EvalReport(pin=pin, metrics={"a": 3.0, "c": 9.0})
        comparison = compare_reports(left, right)
        assert comparison["a"] == {"a": 2.0, "b": 3.0, "delta": 1.0, "relative": 0.5}
        # A key present on one side only appears without deltas, not dropped.
        assert comparison["b"] == {"a": 1.0}
        assert comparison["c"] == {"b": 9.0}

    def test_a_zero_baseline_gives_an_infinite_relative_change(self) -> None:
        pin = EvalPin(steps=30)
        comparison = compare_reports(
            EvalReport(pin=pin, metrics={"a": 0.0}),
            EvalReport(pin=pin, metrics={"a": 1.0}),
        )
        assert comparison["a"]["relative"] == float("inf")

    def test_format_comparison_renders_every_row(self) -> None:
        pin = EvalPin(steps=30)
        rendered = format_comparison(
            compare_reports(
                EvalReport(pin=pin, metrics={"m/a": 1.0, "m/b": 2.0}),
                EvalReport(pin=pin, metrics={"m/a": 1.5}),
            ),
            left="before",
            right="after",
        )
        assert "m/a" in rendered and "m/b" in rendered
        assert "before" in rendered and "after" in rendered
        assert format_comparison({}) == "no metrics in common"

    def test_the_metric_parameters_are_part_of_the_pin(self) -> None:
        # A threshold changed in a metric constructor is a changed metric.
        base = EvalPin(steps=30, metrics={"static_frames": "threshold=0.02"})
        other = replace(base, metrics={"static_frames": "threshold=0.10"})
        with pytest.raises(IncomparableReports, match=r"metrics"):
            compare_reports(EvalReport(pin=base), EvalReport(pin=other))


class TestEvalReport:
    """A report has to survive a JSON round-trip without losing its pin."""

    def test_it_is_json_serialisable(self) -> None:
        report = EvalReport(
            pin=EvalPin(steps=30, metrics={"a": "default"}),
            metrics={"m/x": 1.0},
            skipped={"fvd": "backend missing"},
            external={"human": 4.2},
            run_name="run",
            step=1000,
            notes="why",
        )
        payload = json.dumps(report.to_dict())
        assert json.loads(payload)["pin"]["steps"] == 30

    def test_it_round_trips_through_to_dict(self) -> None:
        report = EvalReport(
            pin=EvalPin(checkpoint="c", steps=40, guidance=3.0, metrics={"a": "b"}),
            metrics={"m/x": 1.0},
            skipped={"fvd": "no backend"},
            external={"human": 4.2},
            run_name="run",
            step=1000,
            notes="why",
        )
        restored = EvalReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert restored.pin == report.pin
        assert restored.metrics == report.metrics
        assert restored.skipped == report.skipped
        assert restored.external == report.external
        assert restored.created_at == report.created_at
        assert restored.notes == report.notes

    def test_it_round_trips_through_a_file(self, tmp_path: Path) -> None:
        report = EvalReport(pin=EvalPin(steps=25), metrics={"m/x": 0.5})
        written = report.save(tmp_path / "nested" / "report.json")
        assert written.is_file()
        assert EvalReport.load(written).pin == report.pin

    def test_a_newer_report_version_is_refused(self) -> None:
        # An unknown pinned field would be silently ignored, and ignoring a
        # pinned field is precisely the failure the pin exists to prevent.
        with pytest.raises(ValueError, match=r"newer than this avgen understands"):
            EvalReport.from_dict({"report_version": 99, "pin": {}})

    def test_an_unreadable_file_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match=r"cannot read evaluation report"):
            EvalReport.load(bad)
        with pytest.raises(ValueError, match=r"cannot read evaluation report"):
            EvalReport.load(tmp_path / "absent.json")

    def test_the_timestamp_and_environment_are_stamped(self) -> None:
        report = EvalReport(pin=EvalPin())
        assert report.created_at
        assert report.environment["python"]
        assert "torch" in report.environment

    def test_capture_environment_never_raises(self) -> None:
        environment = capture_environment()
        assert environment["python"]
        assert environment["platform"]

    def test_with_metrics_returns_a_new_report(self) -> None:
        report = EvalReport(pin=EvalPin(), metrics={"a": 1.0})
        merged = report.with_metrics({"b": 2.0})
        assert merged.metrics == {"a": 1.0, "b": 2.0}
        assert report.metrics == {"a": 1.0}  # the original is untouched

    def test_render_puts_the_settings_above_the_numbers(self) -> None:
        # The order is deliberate: the settings are what make the numbers mean
        # anything.
        rendered = EvalReport(
            pin=EvalPin(checkpoint="step-1", steps=30),
            metrics={"m/x": 1.0},
            skipped={"fvd": "no backend"},
            external={"human": 4.0},
            run_name="run",
            step=1,
        ).render()
        assert rendered.index("settings") < rendered.index("METRICS")
        assert "step-1" in rendered
        assert "SKIPPED" in rendered
        assert "EXTERNAL" in rendered

    def test_render_says_so_when_nothing_was_computed(self) -> None:
        assert "(none computed)" in EvalReport(pin=EvalPin()).render()
        assert "(unnamed run)" in EvalReport(pin=EvalPin()).render()
        assert "(unspecified)" in EvalReport(pin=EvalPin()).render()


class TestBuildMetrics:
    """What will run, what will not, and why — decided once, up front."""

    def test_a_pixel_only_metric_is_skipped_in_latent_space(self) -> None:
        # Its value on latents would not measure what its name says.
        bundle = build_metrics(["saturation", "flicker_index"], decoded=False)
        assert "saturation" not in bundle.metrics
        assert "pixel-space metric" in bundle.skipped["saturation"]
        assert "flicker_index" in bundle.metrics

    def test_a_pixel_only_metric_runs_when_decoding(self) -> None:
        bundle = build_metrics(["saturation"], decoded=True)
        assert "saturation" in bundle.metrics
        assert not bundle.skipped

    def test_an_unknown_metric_name_is_fatal(self) -> None:
        # An unknown name is a typo in a config; running without it is the
        # silent-omission failure the whole subsystem exists to prevent.
        with pytest.raises(MetricError, match=r"unknown metric 'tempral_consistency'"):
            build_metrics(["tempral_consistency"], decoded=False)

    def test_constructor_options_are_recorded_in_the_parameters(self) -> None:
        bundle = build_metrics(
            ["static_frames"],
            decoded=False,
            options={"static_frames": {"threshold": 0.5}},
        )
        assert bundle.parameters["static_frames"] == "threshold=0.5"

    def test_metrics_without_options_record_the_default_marker(self) -> None:
        bundle = build_metrics(["flicker_index"], decoded=False)
        assert bundle.parameters["flicker_index"] == "default"

    @pytest.mark.skipif(
        importlib.util.find_spec("transformers") is not None,
        reason="the transformers backend is installed, so it is not gated here",
    )
    def test_a_gated_metric_lands_in_skipped_not_in_metrics(self) -> None:
        bundle = build_metrics(["clip_score", "flicker_index"], decoded=False)
        assert "clip_score" not in bundle.metrics
        assert "transformers" in bundle.skipped["clip_score"]

    def test_an_empty_bundle_is_legal(self) -> None:
        bundle = MetricBundle()
        assert bundle.metrics == {} and bundle.skipped == {}


class TestShardPrompts:
    """Sharding is strided, and only on the data rank.

    Prompt files are grouped by category, so a contiguous split hands rank 0
    every landscape and rank 3 every portrait — every per-rank diagnostic then
    compares categories rather than ranks.
    """

    def test_a_strided_split_covers_the_set_exactly_once(self) -> None:
        prompts = [f"p{index}" for index in range(10)]
        shards = [shard_prompts(prompts, data_rank=r, data_world=4) for r in range(4)]
        assert sorted(item for shard in shards for item in shard) == sorted(prompts)

    def test_the_split_is_strided_not_contiguous(self) -> None:
        prompts = [f"p{index}" for index in range(8)]
        assert shard_prompts(prompts, data_rank=0, data_world=4) == ("p0", "p4")

    def test_a_world_of_one_returns_everything(self) -> None:
        prompts = ["a", "b", "c"]
        assert shard_prompts(prompts, data_rank=0, data_world=1) == ("a", "b", "c")

    def test_an_indivisible_count_leaves_ranks_uneven(self) -> None:
        # Which is exactly why reduction must be over sufficient statistics.
        prompts = [f"p{index}" for index in range(7)]
        sizes = [
            len(shard_prompts(prompts, data_rank=r, data_world=4)) for r in range(4)
        ]
        assert sizes == [2, 2, 2, 1]

    def test_inconsistent_coordinates_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"data_world must be >= 1"):
            shard_prompts(["a"], data_rank=0, data_world=0)
        with pytest.raises(ValueError, match=r"data_rank must be in \[0, 2\)"):
            shard_prompts(["a"], data_rank=2, data_world=2)


class TestRunEvalSuite:
    """The suite must not let a caller record a pin that does not match the data."""

    def test_the_sample_count_comes_from_what_was_seen(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=random_video()), EvalBatch(video=random_video(seed=1))],
            metrics=["flicker_index"],
            pin=EvalPin(steps=30, num_samples=99999),
        )
        assert report.pin.num_samples == 2 * BATCH

    def test_the_prompt_digest_comes_from_the_global_prompt_set(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=random_video())],
            metrics=["flicker_index"],
            pin=EvalPin(steps=30),
            prompts=["a", "b", "c"],
        )
        assert report.pin.prompt_digest == prompt_set_digest(["a", "b", "c"])

    def test_the_decode_flag_is_forced_into_the_pin(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=random_video())],
            metrics=["flicker_index"],
            pin=EvalPin(steps=30, decoded=False),
            decoded=True,
        )
        assert report.pin.decoded is True

    def test_the_metric_configuration_is_recorded_in_the_pin(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=random_video())],
            metrics=["static_frames"],
            pin=EvalPin(steps=30),
            metric_options={"static_frames": {"threshold": 0.5}},
        )
        assert report.pin.metrics["static_frames"] == "threshold=0.5"

    def test_a_metric_whose_input_was_never_produced_is_recorded_once(self) -> None:
        # An absent row reads as "not measured"; this makes it say "could not
        # be measured, here is why".
        report = run_eval_suite(
            [EvalBatch(video=random_video()), EvalBatch(video=random_video(seed=1))],
            metrics=["av_sync_proxy", "flicker_index"],
            pin=EvalPin(steps=30),
        )
        assert "av_sync_proxy" in report.skipped
        assert "audio" in report.skipped["av_sync_proxy"]
        assert any(key.startswith("flicker_index/") for key in report.metrics)

    def test_a_metric_that_produced_no_statistics_is_recorded(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=torch.randn(1, 3, 1, 8, 8))],
            metrics=["motion_magnitude"],
            pin=EvalPin(steps=30),
        )
        assert "too short" in report.skipped["motion_magnitude"]
        assert report.metrics == {}

    def test_no_batches_at_all_skips_everything(self) -> None:
        report = run_eval_suite([], metrics=["flicker_index"], pin=EvalPin(steps=30))
        assert (
            report.skipped["flicker_index"] == "no batch supplied its required inputs"
        )
        assert report.pin.num_samples == 0

    def test_audio_and_conditional_inputs_reach_their_metrics(self) -> None:
        mask = torch.zeros(BATCH, 1, FRAMES, HEIGHT, WIDTH)
        mask[:, :, :, 2:6, 2:6] = 1.0
        video = random_video()
        report = run_eval_suite(
            [
                EvalBatch(
                    video=video,
                    audio=torch.randn(BATCH, 2, 64),
                    reference=video.clone(),
                    mask=mask,
                    prompts=("a", "b"),
                )
            ],
            metrics=[
                "av_sync_proxy",
                "audio_silence",
                "first_frame_fidelity",
                "inpaint_boundary",
            ],
            pin=EvalPin(steps=30),
        )
        assert report.skipped == {}
        for name in ("av_sync_proxy", "audio_silence", "first_frame_fidelity"):
            assert any(key.startswith(f"{name}/") for key in report.metrics)

    def test_as_inputs_drops_absent_streams(self) -> None:
        batch = EvalBatch(video=random_video(), extra={"context_frames": 3})
        supplied = batch.as_inputs()
        assert set(supplied) == {"video", "context_frames"}
        assert "audio" not in supplied

    def test_a_prebuilt_bundle_is_used_as_given(self) -> None:
        bundle = build_metrics(["flicker_index"], decoded=False)
        report = run_eval_suite(
            [EvalBatch(video=random_video())], metrics=bundle, pin=EvalPin(steps=30)
        )
        assert any(key.startswith("flicker_index/") for key in report.metrics)

    def test_run_name_step_and_notes_are_carried(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=random_video())],
            metrics=["flicker_index"],
            pin=EvalPin(steps=30),
            run_name="my-run",
            step=40000,
            notes="pilot",
        )
        assert (report.run_name, report.step, report.notes) == (
            "my-run",
            40000,
            "pilot",
        )

    def test_two_suite_runs_under_one_pin_compare(self) -> None:
        # The end-to-end promise: same settings in, comparable numbers out.
        pin = EvalPin(steps=30, guidance=5.0, seed=0, frames=FRAMES)
        first = run_eval_suite(
            [EvalBatch(video=random_video())],
            metrics=["flicker_index"],
            pin=pin,
            prompts=["a", "b"],
        )
        second = run_eval_suite(
            [EvalBatch(video=random_video(seed=9))],
            metrics=["flicker_index"],
            pin=replace(pin, checkpoint="later"),
            prompts=["a", "b"],
        )
        assert compare_reports(first, second)


class TestDistributedGatherSemantics:
    """Cross-rank merging is where an evaluation number silently goes wrong.

    Both properties below were broken once. A biased metric or a wrong sample
    count is not a crash; it is a number that gets written into a report and
    believed, so they are asserted directly rather than left to a review.
    """

    @staticmethod
    def _two_rank_gather(payload: Any) -> list[Any]:
        """Simulate a 2-rank all-gather that deserialises every entry.

        ``torch.distributed`` returns a *deserialised copy* for the local
        rank's own entry, not the object that was passed in — so a merge that
        skips its own slot by identity never fires and double-counts itself.
        """
        if isinstance(payload, dict):
            remote = {
                name: {key: (0.0, count) for key, (_, count) in state.items()}
                for name, state in payload.items()
            }
            return [copy.deepcopy(payload), remote]
        return [payload, payload]

    def test_a_gather_does_not_double_count_the_local_rank(self) -> None:
        report = run_eval_suite(
            [EvalBatch(video=static_video())],  # every pair scores cosine 1.0
            metrics=["temporal_consistency"],
            pin=EvalPin(steps=30),
            gather=self._two_rank_gather,
        )
        # One rank at 1.0, one rank at 0.0, equal counts -> 0.5.
        assert report.metrics["temporal_consistency/cosine"] == pytest.approx(
            0.5, abs=1e-4
        )

    def test_a_destination_only_gather_leaves_other_ranks_local(self) -> None:
        """``dist.gather_object`` returns ``None`` off the destination rank.

        Those ranks must fall back to their own statistics and their own sample
        count rather than dying on ``None`` — only the destination reports, but
        every rank has to survive the call to reach the next collective.
        """
        report = run_eval_suite(
            [EvalBatch(video=random_video())],
            metrics=["flicker_index"],
            pin=EvalPin(steps=30),
            gather=lambda _payload: None,  # what gather_object returns off dst
        )
        # BATCH local samples, not a crash and not a cross-rank total.
        assert report.pin.num_samples == BATCH
        assert report.metrics
