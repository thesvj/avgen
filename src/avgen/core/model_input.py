"""The single model ABI: what a denoiser consumes and what it returns.

Every model in avgen has exactly one forward signature::

    def forward(self, inputs: ModelInput) -> ModelOutput: ...

Training builds a :class:`ModelInput` from a batch plus an objective; inference
builds one from a sampler state. Because both paths converge on the same
structure, there is no separate inference code path to drift out of sync with
training — the most common and most expensive class of bug in generative model
codebases.

The input is sequence-first: the generative modalities arrive as
:class:`~avgen.core.tokens.TokenStream` objects that have already been
patchified, so a model never reshapes a grid and the context-parallel layer has
a well-defined axis to shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch.utils import _pytree

from avgen.core._validate import require_dtype, require_shape
from avgen.core.batch import ConditionMode
from avgen.core.tokens import TextContext, TokenStream

__all__ = ["ModelInput", "ModelOutput"]


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Noisy token streams plus conditioning, as a model sees them.

    Args:
        video: Noisy video tokens. Always present; a video model is the point.
        audio: Noisy audio tokens. May be a zero-length stream, in which case
            every audio path in the model degenerates to a no-op.
        text: Frozen text-encoder context. May be empty for an unconditional
            model.
        condition_mode: ``(batch,)`` :class:`ConditionMode` per sample.
            Defaults to ``JOINT``.
    """

    video: TokenStream
    audio: TokenStream
    text: TextContext
    condition_mode: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )

    def __post_init__(self) -> None:
        """Materialise an omitted condition mode as ``JOINT``."""
        if self.condition_mode.numel() == 0:
            object.__setattr__(
                self,
                "condition_mode",
                torch.full(
                    (self.video.batch_size,),
                    int(ConditionMode.JOINT),
                    dtype=torch.int64,
                    device=self.video.device,
                ),
            )

    @property
    def batch_size(self) -> int:
        """Number of samples."""
        return self.video.batch_size

    @property
    def device(self) -> torch.device:
        """Device the tensors live on."""
        return self.video.device

    @property
    def has_audio(self) -> bool:
        """Whether an audio stream is present."""
        return self.audio.length > 0

    @property
    def has_text(self) -> bool:
        """Whether text conditioning is present."""
        return not self.text.is_empty

    @property
    def total_tokens(self) -> int:
        """Generative tokens per sample held by this rank."""
        return self.video.length + self.audio.length

    def replace_streams(
        self,
        *,
        video: TokenStream | None = None,
        audio: TokenStream | None = None,
    ) -> ModelInput:
        """Return a copy with one or both generative streams replaced.

        Args:
            video: Replacement video stream.
            audio: Replacement audio stream.

        Returns:
            The updated input.
        """
        return replace(
            self,
            video=self.video if video is None else video,
            audio=self.audio if audio is None else audio,
        )

    def unconditional(self) -> ModelInput:
        """Return the classifier-free-guidance null branch.

        Only the text context is nulled. Structural conditioning — a clean first
        frame, a temporal prefix — is deliberately preserved, because dropping
        it would make the null branch a different *task* rather than the same
        task without a prompt, and the guidance difference would then be
        meaningless.

        Returns:
            An input with null text conditioning.
        """
        return replace(self, text=self.text.nullified())

    def validate(self) -> None:
        """Validate every stream and the batch-dimension agreement between them.

        Called at the boundaries of training and sampling, never inside a
        compiled region.

        Raises:
            ValueError: On a shape, device, batch-agreement, or condition-mode
                violation.
            TypeError: On a dtype violation.
        """
        self.video.validate()
        self.audio.validate()
        self.text.validate()
        batch = self.video.batch_size
        for name, observed in (
            ("audio", self.audio.batch_size),
            ("text", self.text.features.shape[0]),
        ):
            if observed != batch:
                raise ValueError(
                    f"{name} batch dimension must match video ({batch}); got {observed}"
                )
        require_shape("condition_mode", self.condition_mode, (batch,))
        require_dtype("condition_mode", self.condition_mode, torch.int64)
        valid = torch.zeros_like(self.condition_mode, dtype=torch.bool)
        for mode in ConditionMode:
            valid |= self.condition_mode == int(mode)
        if not bool(valid.all()):
            raise ValueError("condition_mode contains unsupported values")
        device = self.video.device
        for name, tensor in (
            ("audio tokens", self.audio.tokens),
            ("text features", self.text.features),
            ("condition_mode", self.condition_mode),
        ):
            if tensor.device != device:
                raise ValueError(
                    f"{name} must be on video device {device}; got {tensor.device}"
                )


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Predicted per-token fields returned by a model.

    Predictions stay in token space rather than being unpatchified inside the
    model. The objective compares them against a token-space target, so
    unfolding to a grid would be pure waste in the hot loop; the sampler unfolds
    once, at the end.

    Args:
        video: ``(batch, video_length, patch_dim)`` predicted video field.
        audio: ``(batch, audio_length, patch_dim)`` predicted audio field.
        auxiliary: Optional extra predictions — an auxiliary head, a routing
            statistic, a per-token uncertainty — that an objective may consume.
            Kept as an unnamed tuple so a research head can be added without
            changing this dataclass or the checkpoint ABI.
    """

    video: torch.Tensor
    audio: torch.Tensor
    auxiliary: tuple[torch.Tensor, ...] = ()

    def validate(self, inputs: ModelInput) -> None:
        """Validate predicted shapes, dtypes, and device against the input.

        Args:
            inputs: The input this output was produced from.

        Raises:
            ValueError: On a shape or device violation.
            TypeError: If any prediction is not floating point.
        """
        expected_video = (
            inputs.video.batch_size,
            inputs.video.length,
            inputs.video.layout.patch_dim,
        )
        expected_audio = (
            inputs.audio.batch_size,
            inputs.audio.length,
            inputs.audio.layout.patch_dim,
        )
        if tuple(self.video.shape) != expected_video:
            raise ValueError(
                f"video output shape must be {expected_video}; "
                f"got {tuple(self.video.shape)}"
            )
        if inputs.has_audio and tuple(self.audio.shape) != expected_audio:
            raise ValueError(
                f"audio output shape must be {expected_audio}; "
                f"got {tuple(self.audio.shape)}"
            )
        device = inputs.device
        for name, tensor in (
            ("video", self.video),
            ("audio", self.audio),
            *((f"auxiliary[{i}]", t) for i, t in enumerate(self.auxiliary)),
        ):
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point; got {tensor.dtype}")
            if tensor.device != device:
                raise ValueError(
                    f"{name} must be on input device {device}; got {tensor.device}"
                )


_pytree.register_dataclass(ModelInput, serialized_type_name="avgen.ModelInput")
_pytree.register_dataclass(ModelOutput, serialized_type_name="avgen.ModelOutput")
