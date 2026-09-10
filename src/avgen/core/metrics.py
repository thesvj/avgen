"""Scalar metrics produced by one training step.

Every field is a detached zero-dimensional tensor rather than a Python float.
That is deliberate: reading ``.item()`` on a CUDA tensor forces a device
synchronisation, and doing it once per step per metric serialises the pipeline
behind the slowest rank. Keeping metrics on device lets the trainer batch the
reduction and pay for exactly one synchronisation, at logging cadence, not at
step cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch.utils import _pytree

__all__ = ["StepMetrics"]


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Detached scalar tensors returned by a training step.

    Args:
        loss: Total objective value.
        video_loss: Video-modality component before weighting.
        audio_loss: Audio-modality component before weighting.
        valid_video_tokens: Unpadded video tokens in the microbatch.
        valid_audio_tokens: Unpadded audio tokens in the microbatch.
        grad_norm: Global gradient norm at the optimizer step, or zero on a
            gradient-accumulation microbatch that did not step.
        nonfinite: Whether the step was rejected for a non-finite loss or
            gradient.
        skipped: Whether the optimizer step was skipped for any reason. A
            handful of skipped steps in a long run is survivable; a rising rate
            is the earliest signal that a run is diverging.
    """

    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor
    valid_video_tokens: torch.Tensor
    valid_audio_tokens: torch.Tensor
    grad_norm: torch.Tensor
    nonfinite: torch.Tensor
    skipped: torch.Tensor

    def validate(self) -> None:
        """Validate scalar structure, dtypes, device, and detachment.

        Raises:
            ValueError: If any field is non-scalar, still attached to the graph,
                or on a different device from ``loss``.
            TypeError: If any field has the wrong dtype.
        """
        for spec in fields(self):
            value: torch.Tensor = getattr(self, spec.name)
            if value.ndim != 0:
                raise ValueError(
                    f"{spec.name} must be scalar; got shape {tuple(value.shape)}"
                )
            if value.requires_grad:
                raise ValueError(f"{spec.name} must be detached")
            if value.device != self.loss.device:
                raise ValueError(
                    f"{spec.name} must be on loss device {self.loss.device}; "
                    f"got {value.device}"
                )
        for name in ("loss", "video_loss", "audio_loss", "grad_norm"):
            value = getattr(self, name)
            if value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32; got {value.dtype}")
        for name in ("valid_video_tokens", "valid_audio_tokens"):
            value = getattr(self, name)
            if value.dtype is not torch.int64:
                raise TypeError(f"{name} must be int64; got {value.dtype}")
        for name in ("nonfinite", "skipped"):
            value = getattr(self, name)
            if value.dtype is not torch.bool:
                raise TypeError(f"{name} must be bool; got {value.dtype}")

    def to_mapping(self) -> dict[str, float]:
        """Return a host-side mapping, synchronising once.

        This is the only place the framework converts step metrics to Python
        scalars. Call it at logging cadence, never every step.

        Returns:
            Field name to float value.
        """
        return {
            spec.name: float(getattr(self, spec.name).item()) for spec in fields(self)
        }

    @classmethod
    def zeros(cls, device: torch.device | str = "cpu") -> StepMetrics:
        """Return an all-zero instance, useful as a reduction identity.

        Args:
            device: Device to allocate on.

        Returns:
            A zeroed metrics record.
        """
        target = torch.device(device)

        def _scalar(dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros((), dtype=dtype, device=target)

        return cls(
            loss=_scalar(torch.float32),
            video_loss=_scalar(torch.float32),
            audio_loss=_scalar(torch.float32),
            valid_video_tokens=_scalar(torch.int64),
            valid_audio_tokens=_scalar(torch.int64),
            grad_norm=_scalar(torch.float32),
            nonfinite=_scalar(torch.bool),
            skipped=_scalar(torch.bool),
        )


_pytree.register_dataclass(StepMetrics, serialized_type_name="avgen.StepMetrics")
