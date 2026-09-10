"""Dependency-free video metrics, computed on latents or decoded frames.

None of these is a perceptual quality score, and each docstring says plainly
what it does *not* measure. That honesty is the point of the module. The
learned metrics people quote — FVD, CLIPScore — need weights, a network, and a
GPU; they are gated behind :mod:`avgen.eval.learned` and are the right thing to
report in a paper. What these are for is the other job, the one that actually
governs a training run: **catching the failure modes cheaply, every N steps,
with no downloads.**

Those failure modes are specific and every one of them is visible here:

* The model emits a **still image** — :class:`StaticFrameRatio`.
* The model emits **temporally independent noise** — :class:`FlickerIndex`.
* The model emits something that moves but is **incoherent** — high
  :class:`MotionMagnitude` with low :class:`TemporalConsistency`.
* The model emits something coherent but **frozen** — the opposite pair.
* Detail **decays over the clip** — :class:`SharpnessProxy`'s decay ratio, the
  characteristic failure of continuation and autoregressive video models.
* The decoder is **clipping** — :class:`SaturationClipping`.

**Read them in pairs.** Every one of these is trivially maximisable on its own:
a frozen frame scores a perfect temporal consistency, and white noise scores a
maximal motion magnitude. A single number from this module is not evidence of
anything. Consistency *and* motion together are.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from avgen.eval.protocols import RunningMetric, register_metric, require_video

__all__ = [
    "FlickerIndex",
    "MotionMagnitude",
    "SaturationClipping",
    "SharpnessProxy",
    "StaticFrameRatio",
    "TemporalConsistency",
]

#: Guards every normalisation against a constant clip, whose std is exactly
#: zero. Without it a still frame produces NaN and poisons the whole report.
_EPS = 1e-8


def _frame_features(video: torch.Tensor) -> torch.Tensor:
    """Flatten each frame to a feature vector: ``(B, C, T, H, W)`` to ``(B, T, D)``."""
    batch, frames = video.shape[0], video.shape[2]
    return video.permute(0, 2, 1, 3, 4).reshape(batch, frames, -1)


def _per_sample_scale(video: torch.Tensor) -> torch.Tensor:
    """Per-sample standard deviation, used to make metrics scale-invariant.

    Latents and pixels live on completely different scales, and a VAE's latent
    scale changes between checkpoints. Normalising by the clip's own spread is
    what lets one threshold work in both spaces.
    """
    batch = video.shape[0]
    return video.reshape(batch, -1).std(dim=1).clamp_min(_EPS)


@register_metric("temporal_consistency")
class TemporalConsistency(RunningMetric):
    """Cosine similarity between consecutive frames' flattened features.

    **What it measures.** How much a frame resembles the one before it, in raw
    feature space. Computed as the mean over frame pairs of
    ``cos(f_t, f_{t-1})`` where ``f_t`` is the whole frame flattened and
    L2-normalised. A hard cut scores near zero for that pair; a smooth pan
    scores high.

    **What it does not measure.** Anything semantic. It cannot tell a coherent
    tracking shot from a slowly cross-fading slideshow, and — the important one
    — **a completely static video scores a perfect 1.0**. This metric is only
    interpretable next to :class:`MotionMagnitude`. Reported alone it rewards
    exactly the failure mode it looks like it is guarding against.

    The rejected alternative was a learned feature distance (CLIP or DINO
    per-frame embeddings, as most published "subject consistency" metrics use).
    It is a better measure of semantic consistency and it needs a network,
    weights, and a download; it belongs in :mod:`avgen.eval.learned`, not in
    the metric a trainer runs every thousand steps.

    Args:
        stride: Frame gap to compare across. ``1`` is adjacent frames.
            A larger stride measures longer-range drift, which is where
            slow-motion incoherence shows up and adjacent-frame consistency
            does not.
        device: Accumulator device.
    """

    required_inputs = ("video",)

    def __init__(self, *, stride: int = 1, device: torch.device | str = "cpu") -> None:
        super().__init__(device=device)
        if stride < 1:
            raise ValueError(f"stride must be >= 1; got {stride!r}")
        self._stride = stride

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate mean cosine similarity over frame pairs.

        Args:
            **inputs: Must contain ``video``.

        Returns:
            ``cosine`` (mean similarity) and ``worst_pair`` (the minimum over
            pairs, which finds the single hardest cut in the clip).
        """
        video = require_video(inputs["video"], metric=self.name)
        features = _frame_features(video)
        if features.shape[1] <= self._stride:
            return {}
        normalised = features / features.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        similarity = (normalised[:, self._stride :] * normalised[:, : -self._stride]).sum(
            dim=-1
        )
        pairs = int(similarity.numel())
        worst = similarity.min(dim=1).values
        return {
            "cosine": (similarity.sum(), pairs),
            "worst_pair": (worst.sum(), int(worst.numel())),
        }


@register_metric("motion_magnitude")
class MotionMagnitude(RunningMetric):
    """Scale-normalised mean absolute difference between consecutive frames.

    **What it measures.** How much the content changes per frame, expressed in
    units of the clip's own standard deviation so that latents and pixels give
    comparable numbers. It is the direct counter-check on
    :class:`TemporalConsistency`: near zero means the model produced a still
    image regardless of how good the consistency score looks.

    **What it does not measure.** Whether the motion is *correct*, *plausible*,
    or even *spatially coherent*. Independent per-frame noise maximises this
    metric. It is a liveness check, not a quality score.

    A true optical-flow magnitude would measure coherent displacement rather
    than raw change, and would be the better number — it needs a flow network,
    so it is deliberately not here. The frame-difference proxy costs one
    subtraction and catches the failure that matters.

    Args:
        device: Accumulator device.
    """

    required_inputs = ("video",)

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate normalised inter-frame change.

        Args:
            **inputs: Must contain ``video``.

        Returns:
            ``mean`` (average per-frame change) and ``peak`` (the largest
            single-frame change per clip, which separates smooth motion from
            an abrupt scene cut).
        """
        video = require_video(inputs["video"], metric=self.name)
        if video.shape[2] < 2:
            return {}
        scale = _per_sample_scale(video).view(-1, 1)
        deltas = (video[:, :, 1:] - video[:, :, :-1]).abs()
        per_frame = deltas.mean(dim=(1, 3, 4))
        normalised = per_frame / scale
        peak = normalised.max(dim=1).values
        return {
            "mean": (normalised.sum(), int(normalised.numel())),
            "peak": (peak.sum(), int(peak.numel())),
        }


@register_metric("flicker_index")
class FlickerIndex(RunningMetric):
    """Second-difference energy relative to first-difference energy.

    **What it measures.** Whether change between frames is *directional* or
    *oscillating*. For a sequence ``x``, the ratio
    ``E[(x_t - 2x_{t-1} + x_{t-2})^2] / E[(x_t - x_{t-1})^2]`` has three
    reference values worth memorising:

    ==========================  ==========
    Content                     Index
    ==========================  ==========
    Constant-velocity motion     ~0
    Temporally independent noise ~3
    Alternating flicker          >3
    ==========================  ==========

    The value 3 falls out of the variances: for i.i.d. frames of variance
    ``s``, the first difference has variance ``2s`` and the second ``6s``. So a
    reading near 3 means the model is producing frames that are statistically
    independent — that is, it has not learned temporal structure at all, which
    is a very different diagnosis from "the motion is wrong".

    **What it does not measure.** Spatial artefacts, and any flicker whose
    period is longer than two frames. A slow brightness pulse across ten frames
    reads as smooth motion here.

    Args:
        device: Accumulator device.
    """

    required_inputs = ("video",)

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate the second-difference to first-difference energy ratio.

        Args:
            **inputs: Must contain ``video``.

        Returns:
            ``index`` (the ratio) and ``second_difference`` (its numerator,
            normalised by the clip scale, which distinguishes "low ratio
            because it is smooth" from "low ratio because nothing moves").
        """
        video = require_video(inputs["video"], metric=self.name)
        if video.shape[2] < 3:
            return {}
        first = video[:, :, 1:] - video[:, :, :-1]
        second = first[:, :, 1:] - first[:, :, :-1]
        first_energy = first[:, :, 1:].pow(2).flatten(1).mean(dim=1)
        second_energy = second.pow(2).flatten(1).mean(dim=1)
        ratio = second_energy / first_energy.clamp_min(_EPS)
        scale = _per_sample_scale(video).pow(2)
        return {
            "index": (ratio.sum(), int(ratio.numel())),
            "second_difference": (
                (second_energy / scale).sum(),
                int(second_energy.numel()),
            ),
        }


@register_metric("static_frames")
class StaticFrameRatio(RunningMetric):
    """Fraction of clips and frame pairs that barely change at all.

    **What it measures.** The single most common degenerate output of a
    text-to-video model: a still image with a light dusting of noise, produced
    when the model has learned the image prior and not the temporal one. A pair
    of frames counts as static when its normalised mean absolute difference
    falls below ``threshold``; a clip counts as static when *every* pair does.

    **What it does not measure.** Partial freezing. A clip where the subject is
    frozen against a moving background passes, because the background alone
    keeps the frame-level difference above threshold. Detecting that needs
    spatially-local statistics, which is a different metric.

    **The threshold is the load-bearing part.** It is expressed in units of the
    clip's own standard deviation, so it transfers between latent and pixel
    space, but the default of 0.02 was chosen to be conservative — it flags
    only clips that are visually still, and it will miss near-still ones. Tune
    it against your own decoder before quoting the number.

    Args:
        threshold: Normalised per-frame change below which a pair is static.
        device: Accumulator device.
    """

    required_inputs = ("video",)

    def __init__(
        self, *, threshold: float = 0.02, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device=device)
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0; got {threshold!r}")
        self._threshold = threshold

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate static-pair and static-clip rates.

        Args:
            **inputs: Must contain ``video``.

        Returns:
            ``pair_rate`` (fraction of frame pairs that are static) and
            ``clip_rate`` (fraction of clips that are entirely static).
        """
        video = require_video(inputs["video"], metric=self.name)
        if video.shape[2] < 2:
            return {}
        scale = _per_sample_scale(video).view(-1, 1)
        deltas = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=(1, 3, 4))
        normalised = deltas / scale
        static_pairs = (normalised < self._threshold).float()
        static_clips = static_pairs.min(dim=1).values
        return {
            "pair_rate": (static_pairs.sum(), int(static_pairs.numel())),
            "clip_rate": (static_clips.sum(), int(static_clips.numel())),
        }


@register_metric("saturation")
class SaturationClipping(RunningMetric):
    """Fraction of samples pinned at the ends of the value range.

    **What it measures.** Whether the decoder output is being clipped. High
    classifier-free guidance is the usual cause: it pushes predictions outside
    the trained range, the decoder saturates, and the result is the blown-out
    look that reads as "over-guided". A rising clipped fraction across a
    guidance sweep is the quantitative version of that observation.

    **What it does not measure.** Anything, in latent space. Latents have no
    meaningful range, so a clipped fraction computed on them is noise with a
    name. The metric declares :attr:`pixel_space_only` and the suite skips it
    with a note rather than reporting a number that looks like data.

    Args:
        value_range: The ``(low, high)`` interval the decoder is expected to
            produce. ``(-1, 1)`` matches every diffusion decoder avgen ships.
        tolerance: Distance from an endpoint that still counts as clipped, as
            a fraction of the range. Exact equality misses values that came
            back at 0.9997 from a bilinear resize.
        device: Accumulator device.
    """

    required_inputs = ("video",)
    pixel_space_only = True

    def __init__(
        self,
        *,
        value_range: tuple[float, float] = (-1.0, 1.0),
        tolerance: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        low, high = value_range
        if not high > low:
            raise ValueError(
                f"value_range must be increasing; got {value_range!r}"
            )
        self._low = low
        self._high = high
        self._margin = tolerance * (high - low)

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate the clipped fraction and the out-of-range fraction.

        Args:
            **inputs: Must contain ``video``.

        Returns:
            ``clipped_fraction`` (values within tolerance of an endpoint) and
            ``out_of_range_fraction`` (values strictly outside, which means the
            caller did not clamp and the number is worse than it looks).
        """
        video = require_video(inputs["video"], metric=self.name)
        elements = int(video.numel())
        at_low = video <= self._low + self._margin
        at_high = video >= self._high - self._margin
        outside = (video < self._low) | (video > self._high)
        return {
            "clipped_fraction": ((at_low | at_high).float().sum(), elements),
            "out_of_range_fraction": (outside.float().sum(), elements),
        }


#: A 3x3 discrete Laplacian. Chosen over a Sobel pair because sharpness is an
#: isotropic property and a single symmetric kernel avoids the two-pass
#: gradient-magnitude combination for the same signal.
_LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)


@register_metric("sharpness")
class SharpnessProxy(RunningMetric):
    """Laplacian response variance per frame, plus its decay across the clip.

    **What it measures.** High-spatial-frequency energy — the classic
    "variance of Laplacian" focus measure. Its absolute value is not
    comparable across resolutions or across latent spaces, so the number that
    earns its place is ``decay``: the mean sharpness of the last third of the
    clip divided by that of the first third.

    ``decay`` below 1 is the signature failure of continuation, autoregressive,
    and long-video-by-extension models — each generated chunk conditions on a
    slightly blurrier one, and detail bleeds away monotonically. It is obvious
    in a side-by-side and invisible in an FVD score, which is exactly why it
    belongs in a cheap always-on metric.

    **What it does not measure.** Perceptual sharpness. Additive
    high-frequency noise raises this score, so a noisy output beats a clean
    one. Read it with :class:`FlickerIndex`, which catches that case.

    Args:
        device: Accumulator device.
    """

    required_inputs = ("video",)

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate mean sharpness and the first-third to last-third ratio.

        Args:
            **inputs: Must contain ``video``.

        Returns:
            ``mean`` (scale-normalised Laplacian variance) and, when the clip
            has at least three frames, ``decay`` (last third over first third).
        """
        video = require_video(inputs["video"], metric=self.name)
        batch, channels, frames, height, width = video.shape
        if height < 3 or width < 3:
            # A 3x3 kernel on a 2-pixel axis measures the padding, not the image.
            return {}
        kernel = _LAPLACIAN.to(video.device).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        response = torch.nn.functional.conv2d(flat, kernel, groups=channels)
        per_frame = response.flatten(1).var(dim=1).view(batch, frames)
        scale = _per_sample_scale(video).pow(2).view(-1, 1)
        normalised = per_frame / scale

        results: dict[str, tuple[torch.Tensor, int]] = {
            "mean": (normalised.sum(), int(normalised.numel()))
        }
        third = frames // 3
        if third >= 1:
            head = normalised[:, :third].mean(dim=1)
            tail = normalised[:, -third:].mean(dim=1)
            decay = tail / head.clamp_min(_EPS)
            results["decay"] = (decay.sum(), int(decay.numel()))
        return results
