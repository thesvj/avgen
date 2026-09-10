"""CLIPScore, the one gated metric avgen implements end to end.

Kept in a private module so that :mod:`avgen.eval.learned` can *declare* the
metric without importing ``transformers`` at module scope — the declaration is
what ``avgen info`` prints, and it must be free on a machine with no optional
extras installed.

CLIPScore is prompt-image agreement: the cosine similarity between the text
embedding of the prompt and the image embedding of each frame, averaged over
frames. Its well-known weaknesses are stated in the class docstring rather than
in a footnote, because the number is quoted far more often than it is
understood.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from avgen.eval.protocols import MetricError, RunningMetric, require_video

__all__ = ["CLIPScore"]


class CLIPScore(RunningMetric):
    """Cosine similarity between prompt embeddings and per-frame image embeddings.

    **What it measures.** Whether the generated frames contain what the prompt
    asked for, according to CLIP. Averaged over frames, so it is an image-level
    metric applied to a video.

    **What it does not measure.** Anything temporal — a shuffled clip scores
    identically. Anything compositional — CLIP is famously weak at binding
    attributes to objects, so "a red cube on a blue sphere" and its swap score
    nearly the same. And it saturates: above roughly 0.31 with the standard
    ViT-L/14 checkpoint, differences stop tracking anything a person would
    notice, which is precisely the range every competent model now occupies.

    **The number depends on the checkpoint.** ViT-B/32 and ViT-L/14 produce
    different values for the same video, so the model id is recorded in the
    report's pin and two reports with different ids will refuse to compare.

    Args:
        model_id: A ``transformers`` CLIP checkpoint identifier.
        device: Device the CLIP tower runs on. The metric's accumulators stay
            on CPU regardless; only the tower moves.
        frame_stride: Evaluate every n-th frame. CLIP over every frame of a
            long clip dominates evaluation time and adds almost nothing, since
            adjacent frames embed nearly identically.

    Raises:
        RuntimeError: If ``transformers`` is not installed. The message names
            the extra.
    """

    name = "clip_score"
    required_inputs = ("video", "prompts")
    pixel_space_only = True

    def __init__(
        self,
        *,
        model_id: str = "openai/clip-vit-large-patch14",
        device: torch.device | str = "cpu",
        frame_stride: int = 1,
    ) -> None:
        super().__init__(device="cpu")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1; got {frame_stride!r}")
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "clip_score needs transformers; install it with "
                "pip install 'avgen[text]'"
            ) from error

        self.model_id = model_id
        self._stride = frame_stride
        self._tower_device = torch.device(device)
        self._model = CLIPModel.from_pretrained(model_id).eval().to(self._tower_device)
        self._processor = CLIPProcessor.from_pretrained(model_id)

    def fingerprint(self) -> str:
        """Return the identifier recorded in an evaluation report's pin.

        Returns:
            The checkpoint id and frame stride, which together determine the
            number this metric produces.
        """
        return f"clip:{self.model_id}:stride{self._stride}"

    @torch.no_grad()
    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate mean and minimum per-frame prompt agreement.

        Args:
            **inputs: Must contain ``video`` (pixels in ``[-1, 1]``) and
                ``prompts`` (one string per batch element).

        Returns:
            ``mean`` (average over frames) and ``worst_frame`` (the minimum
            over frames, which finds the moment the video drifts off-prompt).

        Raises:
            MetricError: If the prompt count does not match the batch size.
        """
        video = require_video(inputs["video"], metric=self.name)
        prompts: Sequence[str] = inputs["prompts"]
        batch, _, frames = video.shape[0], video.shape[1], video.shape[2]
        if len(prompts) != batch:
            raise MetricError(
                f"{self.name}: got {len(prompts)} prompts for a batch of {batch}"
            )

        indices = list(range(0, frames, self._stride))
        # CLIP expects [0, 1] pixels; the framework's decoders emit [-1, 1].
        selected = video[:, :, indices].clamp(-1.0, 1.0).add(1.0).mul(0.5)
        flat = selected.permute(0, 2, 1, 3, 4).reshape(-1, *selected.shape[-3:])

        text = self._processor(
            text=list(prompts), return_tensors="pt", padding=True, truncation=True
        )
        text = {key: value.to(self._tower_device) for key, value in text.items()}
        text_features = self._model.get_text_features(**text)
        image_features = self._model.get_image_features(
            pixel_values=flat.to(self._tower_device)
        )

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(batch, len(indices), -1)
        similarity = (image_features * text_features.unsqueeze(1)).sum(dim=-1)

        return {
            "mean": (similarity.mean(dim=1).sum().cpu(), batch),
            "worst_frame": (similarity.min(dim=1).values.sum().cpu(), batch),
        }
