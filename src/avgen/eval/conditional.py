"""Metrics for conditional generation, where a correct answer partly exists.

Unconditional text-to-video has no ground truth, which is why its evaluation is
so hard. The conditional modes do not have that excuse — image-to-video is
*given* the first frame, continuation is *given* the preceding clip, inpainting
is *given* everything outside the mask — and so each admits a metric that is
close to an objective correctness check.

These are the metrics that catch the specific bugs of each mode:

* **I2V that ignores its conditioning.** The model produces a plausible video
  of something else. First-frame fidelity is near-binary here: a working
  implementation reproduces the conditioning frame almost exactly, a broken one
  does not come close, and there is very little in between.
* **Continuation with a visible seam.** The join between context and generation
  is a hard cut. Measured as a *ratio* against the natural frame-to-frame
  change on either side, because the absolute discontinuity depends entirely on
  how much motion the clip has.
* **Inpainting that does not meet its boundary.** The generated region is
  internally fine and does not join the surrounding content, producing the
  cut-out look. Measured across the mask edge, again as a ratio against the
  interior.

All three follow the same principle: **normalise against what the content
itself does**, never against an absolute threshold. A high-motion clip has a
large frame-to-frame delta everywhere, and an absolute seam threshold flags it
while missing a real seam in a static one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from avgen.eval.protocols import (
    MetricError,
    RunningMetric,
    register_metric,
    require_video,
)

__all__ = ["FirstFrameFidelity", "InpaintBoundaryConsistency", "SeamContinuity"]

_EPS = 1e-8


@register_metric("first_frame_fidelity")
class FirstFrameFidelity(RunningMetric):
    """Agreement between the generated first frame and the I2V conditioning image.

    **What it measures.** Whether image-to-video conditioning is actually
    reaching the model. Reports mean squared error, PSNR relative to the
    reference frame's own dynamic range, and the cosine similarity of the
    flattened frames.

    Read PSNR as a diagnostic with two regimes rather than a quality score.
    Above roughly 30 dB the conditioning path works and the residual is
    reconstruction error from the autoencoder. Below roughly 15 dB the model is
    not conditioning on the frame at all, whatever the samples look like. The
    middle is rare, and when it happens it usually means the conditioning is
    being applied at the wrong noise level.

    **What it does not measure.** Everything after frame zero. A model that
    reproduces the conditioning frame perfectly and then drifts into unrelated
    content scores perfectly here; pair it with
    :class:`~avgen.eval.video.TemporalConsistency`.

    Args:
        frame_index: Which frame is the conditioned one. Zero for standard
            I2V; a different index for models conditioned on a middle frame.
        device: Accumulator device.
    """

    required_inputs = ("video", "reference")

    def __init__(
        self, *, frame_index: int = 0, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device=device)
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0; got {frame_index!r}")
        self._index = frame_index

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate MSE, PSNR, and cosine similarity on the conditioned frame.

        Args:
            **inputs: Must contain ``video`` and ``reference``.

        Returns:
            ``mse``, ``psnr_db``, and ``cosine``.

        Raises:
            MetricError: If the two tensors disagree in shape, or the frame
                index is out of range.
        """
        video = require_video(inputs["video"], metric=self.name)
        reference = require_video(
            inputs["reference"], metric=self.name, argument="reference"
        )
        if video.shape != reference.shape:
            raise MetricError(
                f"{self.name}: video {tuple(video.shape)} and reference "
                f"{tuple(reference.shape)} must have the same shape"
            )
        if self._index >= video.shape[2]:
            raise MetricError(
                f"{self.name}: frame_index={self._index} is out of range for a "
                f"clip with {video.shape[2]} frames"
            )
        generated = video[:, :, self._index].flatten(1)
        target = reference[:, :, self._index].flatten(1)

        mse = (generated - target).pow(2).mean(dim=1)
        # PSNR against the reference's own range rather than a fixed peak of 1,
        # so the metric is meaningful in latent space too.
        span = (target.max(dim=1).values - target.min(dim=1).values).clamp_min(_EPS)
        psnr = 10.0 * torch.log10(span.pow(2) / mse.clamp_min(_EPS))
        cosine = torch.nn.functional.cosine_similarity(generated, target, dim=1)
        batch = int(mse.numel())
        return {
            "mse": (mse.sum(), batch),
            "psnr_db": (psnr.sum(), batch),
            "cosine": (cosine.sum(), batch),
        }


@register_metric("seam_continuity")
class SeamContinuity(RunningMetric):
    """Discontinuity at a continuation join, relative to the clip's own motion.

    **What it measures.** The frame-to-frame change *across* the boundary
    between conditioning context and generated continuation, divided by the mean
    frame-to-frame change on either side of it. The ratio is the number to read:

    ============  ==================================================
    ``ratio``      Interpretation
    ============  ==================================================
    ~1.0           Seamless — the join looks like ordinary motion.
    2-3            Visible on close inspection.
    > 4            A hard cut.
    < 0.5          The continuation froze at the boundary, which is
                   the *other* failure and looks like over-smoothing.
    ============  ==================================================

    Normalising by the clip's own motion is what makes one threshold work for
    both a static interview shot and a fast pan. An absolute discontinuity
    threshold flags every high-motion clip and misses every seam in a still
    one.

    **What it does not measure.** Semantic continuity across the seam — the
    subject changing identity mid-clip is perfectly smooth by this metric — and
    any drift that accumulates over the continuation rather than appearing at
    the join.

    Args:
        context_frames: Number of leading frames that were supplied as
            conditioning. The seam is between ``context_frames - 1`` and
            ``context_frames``.
        device: Accumulator device.
    """

    required_inputs = ("video",)

    def __init__(
        self, *, context_frames: int = 1, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device=device)
        if context_frames < 1:
            raise ValueError(f"context_frames must be >= 1; got {context_frames!r}")
        self._context = context_frames

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate the seam-to-interior change ratio.

        Args:
            **inputs: Must contain ``video``. ``context_frames`` may be passed
                per-batch to override the constructor value, which is what the
                suite does when the continuation length varies by sample.

        Returns:
            ``ratio`` (seam change over interior change) and ``seam_delta``
            (the raw normalised change at the join).

        Raises:
            MetricError: If the clip is too short to contain a seam with
                interior frames on both sides.
        """
        video = require_video(inputs["video"], metric=self.name)
        context = int(inputs.get("context_frames", self._context))
        frames = video.shape[2]
        if context < 1 or context >= frames:
            raise MetricError(
                f"{self.name}: context_frames={context} leaves no continuation "
                f"in a clip of {frames} frames"
            )
        if frames < 4:
            return {}

        deltas = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=(1, 3, 4))
        seam = deltas[:, context - 1]
        interior = torch.cat([deltas[:, : context - 1], deltas[:, context:]], dim=1)
        if interior.shape[1] == 0:
            return {}
        baseline = interior.mean(dim=1).clamp_min(_EPS)
        scale = video.flatten(1).std(dim=1).clamp_min(_EPS)
        batch = int(seam.numel())
        return {
            "ratio": ((seam / baseline).sum(), batch),
            "seam_delta": ((seam / scale).sum(), batch),
        }


@register_metric("inpaint_boundary")
class InpaintBoundaryConsistency(RunningMetric):
    """Spatial gradient mismatch across an inpainting mask boundary.

    **What it measures.** Whether the generated region joins the preserved one.
    The mask is dilated by one element and eroded by one; the ring between the
    two is the boundary. The metric compares the mean spatial gradient
    magnitude on that ring against the mean inside the generated region and
    inside the preserved region, and reports the ratio. A seamless inpaint has
    a boundary gradient in line with its surroundings — a ratio near 1. The
    cut-out look is a ratio well above 1.

    It also reports ``leakage``: the mean absolute difference between the
    generated output and the reference **outside** the mask, where the model was
    supposed to change nothing. Anything meaningfully above the autoencoder's
    own reconstruction error means the inpainting path is overwriting preserved
    content, which is a correctness bug rather than a quality one and is easy
    to miss by eye.

    **What it does not measure.** Whether the filled content is *right* — a
    perfectly blended wrong object scores perfectly.

    Args:
        device: Accumulator device.
    """

    required_inputs = ("video", "mask")

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate boundary gradient ratio and preserved-region leakage.

        Args:
            **inputs: Must contain ``video`` and ``mask``; ``reference`` enables
                the leakage statistic.

        Returns:
            ``gradient_ratio`` and, when a reference is supplied, ``leakage``.

        Raises:
            MetricError: If the mask does not broadcast against the video.
        """
        video = require_video(inputs["video"], metric=self.name)
        mask = inputs["mask"]
        if not isinstance(mask, torch.Tensor) or mask.ndim != 5:
            raise MetricError(
                f"{self.name}: mask must be a (batch, 1, frames, height, width) "
                f"tensor; got {type(mask).__name__} "
                f"{tuple(getattr(mask, 'shape', ()))}"
            )
        mask = mask.detach().float()
        if mask.shape[0] != video.shape[0] or mask.shape[2:] != video.shape[2:]:
            raise MetricError(
                f"{self.name}: mask {tuple(mask.shape)} does not align with "
                f"video {tuple(video.shape)} on batch or spatial dimensions"
            )

        batch, _, frames, height, width = video.shape
        if height < 3 or width < 3:
            return {}

        # Spatial gradient magnitude, forward differences padded to keep shape.
        grad_y = torch.zeros_like(video)
        grad_x = torch.zeros_like(video)
        grad_y[:, :, :, :-1] = (video[:, :, :, 1:] - video[:, :, :, :-1]).abs()
        grad_x[:, :, :, :, :-1] = (video[:, :, :, :, 1:] - video[:, :, :, :, :-1]).abs()
        gradient = (grad_y + grad_x).mean(dim=1, keepdim=True)

        flat = mask.reshape(batch * frames, 1, height, width)
        dilated = torch.nn.functional.max_pool2d(flat, 3, stride=1, padding=1)
        eroded = -torch.nn.functional.max_pool2d(-flat, 3, stride=1, padding=1)
        ring = (dilated - eroded).reshape(batch, 1, frames, height, width)

        ring_weight = ring.flatten(1).sum(dim=1)
        interior_mask = 1.0 - ring
        interior_weight = interior_mask.flatten(1).sum(dim=1)
        usable = (ring_weight > 0) & (interior_weight > 0)
        if not bool(usable.any()):
            # A mask covering everything or nothing has no boundary to measure.
            return {}

        ring_energy = (gradient * ring).flatten(1).sum(dim=1) / ring_weight.clamp_min(
            _EPS
        )
        interior_energy = (gradient * interior_mask).flatten(1).sum(
            dim=1
        ) / interior_weight.clamp_min(_EPS)
        ratio = (ring_energy / interior_energy.clamp_min(_EPS))[usable]

        results: dict[str, tuple[torch.Tensor, int]] = {
            "gradient_ratio": (ratio.sum(), int(ratio.numel()))
        }

        reference = inputs.get("reference")
        if isinstance(reference, torch.Tensor) and reference.shape == video.shape:
            preserved = 1.0 - mask
            weight = preserved.flatten(1).sum(dim=1).clamp_min(_EPS)
            difference = (
                (video - reference.detach().float()).abs().mean(dim=1, keepdim=True)
            )
            scale = video.flatten(1).std(dim=1).clamp_min(_EPS)
            leakage = (difference * preserved).flatten(1).sum(dim=1) / weight / scale
            results["leakage"] = (leakage.sum(), int(leakage.numel()))
        return results
