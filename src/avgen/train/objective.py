"""Rectified flow matching: the loss the whole framework is built around.

Given a clean latent ``x_0`` and Gaussian noise ``eps``, rectified flow defines
the straight-line interpolant

.. code-block:: text

    x_t = (1 - t) * x_0 + t * eps          t in [0, 1]

and trains the model to predict its constant velocity

.. code-block:: text

    v = d x_t / d t = eps - x_0

Two properties make this the right default for video, and both are about
inference cost rather than training quality:

* **The path is straight.** The exact solution of the probability-flow ODE for
  the interpolant above is a line, so an Euler sampler with very few steps is
  already close to exact. Curved schedules (variance-preserving DDPM) need many
  more steps to follow the same trajectory, and a video sample costs steps times
  the whole sequence.
* **The target is well conditioned at both endpoints.** There is no
  ``1/sqrt(1 - alpha_bar)`` blowing up as ``t`` approaches 1, so no loss
  reweighting is required to keep the gradient scale sane, and fp16/bf16 does
  not overflow at the ends of the schedule.

The subtleties that this module exists to get right are elsewhere:

**Independent per-modality noise.** Video and audio are noised to *different*
levels drawn independently. That is what makes one set of weights cover T2V,
V2A, A2V, and joint generation: at inference, holding audio at ``t = 0`` while
sweeping video from 1 to 0 is a configuration the model has actually trained on,
because the pair ``(t_video, t_audio)`` was sampled over the whole square rather
than only along its diagonal. Tie the two levels together and the off-diagonal
tasks are extrapolation.

**Conditioning tokens are excluded from the loss.** They are handed to the model
clean, so predicting their velocity is predicting a function of the input. A
model rewarded for that learns the identity on anchored tokens, and the failure
is specific and recognisable: an image-to-video model that emits a still frame,
or a continuation model whose extension freezes on the last conditioning frame.
:meth:`~avgen.core.tokens.TokenStream.loss_mask` already encodes the rule; this
module normalises by its count, not by the padded token count.

**Context parallelism changes what "mean" means.** Under CP each rank holds a
slice of the sequence and can only compute its slice's numerator and
denominator. Dividing locally and averaging the results afterwards gives the
mean of per-shard means, which equals the global mean only when every shard has
the same number of supervised tokens — and shards differ whenever conditioning
or padding is not uniform along the sequence, which for image-to-video is
*always* (the anchor lives entirely in the first shard). The reduction here is
therefore over the numerator and denominator separately. See
:meth:`FlowMatchingObjective.__call__` for the accompanying gradient scaling,
which is the half of this that is easy to get wrong silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from avgen.core._validate import require_finite
from avgen.core.batch import MediaBatch
from avgen.core.model_input import ModelInput, ModelOutput
from avgen.core.patchify import GridPatchifier, Patchifier
from avgen.core.rng import RNGStreams
from avgen.core.tokens import TextContext, TokenStream
from avgen.parallel.comm import all_reduce_sum
from avgen.parallel.context import shard_stream
from avgen.train.conditioning import ConditioningPlan, ConditioningSampler
from avgen.train.timestep import TimestepSampler

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingObjective",
    "Objective",
    "ObjectiveOutput",
]

#: Timesteps are clamped away from the endpoints before any reciprocal is taken.
#: The interpolant itself is perfectly well behaved at 0 and 1; the SNR-based
#: weightings are not, and a single inf here poisons the whole batch's loss.
_TIMESTEP_EPS = 1e-5

_WEIGHTING_SCHEMES = ("uniform", "sigma_sqrt", "cosmap")


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    """The scalars a training step needs from one objective evaluation.

    ``loss`` is the only field that carries a gradient. The per-modality
    components are detached reporting values: they are what a human reads to see
    whether audio is learning at all while video dominates the total, and they
    must never be a second path into the backward graph.

    Args:
        loss: Total objective, fp32 scalar, still attached to the graph.
        video_loss: Video component before modality weighting, detached.
        audio_loss: Audio component before modality weighting, detached.
        valid_video_tokens: Unpadded video tokens the loss saw, int64 scalar.
        valid_audio_tokens: Unpadded audio tokens the loss saw, int64 scalar.
    """

    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor
    valid_video_tokens: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )
    valid_audio_tokens: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64)
    )

    def __post_init__(self) -> None:
        """Materialise omitted token counts as zero on the loss device."""
        for name in ("valid_video_tokens", "valid_audio_tokens"):
            value: torch.Tensor = getattr(self, name)
            if value.numel() == 0:
                object.__setattr__(
                    self,
                    name,
                    torch.zeros((), dtype=torch.int64, device=self.loss.device),
                )

    def validate(self) -> None:
        """Validate scalar structure, dtype, device, and detachment.

        Unlike :class:`~avgen.core.metrics.StepMetrics`, ``loss`` is *required*
        to still require grad when the model is training, so detachment is
        checked only on the reporting fields.

        Raises:
            ValueError: If a field is non-scalar, on the wrong device, or a
                reporting field is still attached to the graph.
            TypeError: If a field has the wrong dtype.
        """
        device = self.loss.device
        for name in (
            "loss",
            "video_loss",
            "audio_loss",
            "valid_video_tokens",
            "valid_audio_tokens",
        ):
            value: torch.Tensor = getattr(self, name)
            if value.ndim != 0:
                raise ValueError(
                    f"{name} must be scalar; got shape {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError(
                    f"{name} must be on loss device {device}; got {value.device}"
                )
        for name in ("loss", "video_loss", "audio_loss"):
            value = getattr(self, name)
            if value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32; got {value.dtype}")
        for name in ("video_loss", "audio_loss"):
            if getattr(self, name).requires_grad:
                raise ValueError(f"{name} must be detached; it is a reporting value")
        for name in ("valid_video_tokens", "valid_audio_tokens"):
            value = getattr(self, name)
            if value.dtype is not torch.int64:
                raise TypeError(f"{name} must be int64; got {value.dtype}")


@runtime_checkable
class Objective(Protocol):
    """Turns a batch and a model into a differentiable scalar.

    The objective owns everything between the dense batch and the loss:
    conditioning, noising, patchification, the model call, and the reduction.
    The training step owns only the backward pass, the clip, and the optimizer,
    which is what lets a new research objective be dropped in without a new
    trainer.
    """

    def __call__(
        self,
        model: nn.Module,
        batch: MediaBatch,
        rng: RNGStreams,
        *,
        patchifier: Patchifier,
        cp_mesh: DeviceMesh | None = None,
    ) -> ObjectiveOutput:
        """Evaluate the objective for one microbatch."""
        ...


@dataclass(frozen=True, slots=True)
class FlowMatchingConfig:
    """Everything about the flow-matching loss that a config file can set.

    Args:
        video_weight: Multiplier on the video component of the total.
        audio_weight: Multiplier on the audio component. The two components are
            each a *mean over their own tokens* before weighting, so this is a
            true relative importance and not an accident of token counts — a
            10-second clip has roughly a hundred times more video tokens than
            audio tokens, and pooling both into one mean would let video drown
            audio entirely.
        independent_modality_noise: Whether video and audio draw independent
            noise levels. Leave on unless reproducing a single-modality paper.
        timestep_weighting: Per-timestep loss weight. ``"uniform"`` applies
            none; ``"sigma_sqrt"`` (``1 / t^2``) and ``"cosmap"`` are the SD3
            weightings, which up-weight the low-noise end where the residual
            error is small but perceptually dominant.
        snr_gamma: Optional Min-SNR-gamma clamp. ``None`` disables it. The
            weight is ``min(SNR, gamma) / (SNR + 1)`` with ``SNR = ((1-t)/t)^2``,
            the velocity-prediction form: it stops the near-clean timesteps,
            whose SNR diverges, from dominating the gradient and starving the
            structural timesteps the model actually needs.
        validate: Whether to run the contract validators on the constructed
            model input and output. Cheap, but it is a boundary check, so it is
            off in the steady state and on for the first steps of a run and in
            tests.
    """

    video_weight: float = 1.0
    audio_weight: float = 1.0
    independent_modality_noise: bool = True
    timestep_weighting: str = "uniform"
    snr_gamma: float | None = None
    validate: bool = False

    def __post_init__(self) -> None:
        """Validate weights, the weighting scheme name, and the SNR clamp."""
        for name in ("video_weight", "audio_weight"):
            value = float(getattr(self, name))
            require_finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value!r}")
        if self.timestep_weighting not in _WEIGHTING_SCHEMES:
            options = ", ".join(_WEIGHTING_SCHEMES)
            raise ValueError(
                f"timestep_weighting must be one of {options}; "
                f"got {self.timestep_weighting!r}"
            )
        if self.snr_gamma is not None:
            require_finite("snr_gamma", self.snr_gamma)
            if self.snr_gamma <= 0.0:
                raise ValueError(f"snr_gamma must be positive; got {self.snr_gamma!r}")


class FlowMatchingObjective:
    """Rectified-flow velocity matching over video and audio token streams.

    The alternative that lost was epsilon-prediction with a variance-preserving
    schedule, which is what the original diffusion literature uses. It is not
    worse at convergence, but its sampling trajectory is curved, so it needs
    several times as many inference steps for the same quality — and for video,
    inference steps are the entire cost story.

    Args:
        config: Loss weighting and validation policy.
        timestep_sampler: Draws the per-sample noise levels.
        conditioning_sampler: Draws the task mixture and the clean anchors.
        audio_patchifier: Patchifier for the 1-D audio stream. The video
            patchifier cannot be reused: it folds 2x2 spatial patches, and an
            audio latent has spatial extent 1, which is not divisible by 2.
            Defaults to unit patches, which is the right choice when the audio
            codec already compresses time.
        weight_fn: Optional hook receiving ``(batch,)`` timesteps and returning
            ``(batch,)`` non-negative weights. Composes multiplicatively with
            ``config.timestep_weighting`` and ``config.snr_gamma``, and exists so
            a research weighting can be tried without adding a config enum value
            and a branch.
    """

    __slots__ = (
        "audio_patchifier",
        "conditioning_sampler",
        "config",
        "timestep_sampler",
        "weight_fn",
    )

    def __init__(
        self,
        config: FlowMatchingConfig,
        timestep_sampler: TimestepSampler,
        conditioning_sampler: ConditioningSampler,
        *,
        audio_patchifier: Patchifier | None = None,
        weight_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.timestep_sampler = timestep_sampler
        self.conditioning_sampler = conditioning_sampler
        self.audio_patchifier = audio_patchifier or GridPatchifier(
            patch_frames=1, patch_height=1, patch_width=1
        )
        self.weight_fn = weight_fn

    def __call__(
        self,
        model: nn.Module,
        batch: MediaBatch,
        rng: RNGStreams,
        *,
        patchifier: Patchifier,
        cp_mesh: DeviceMesh | None = None,
    ) -> ObjectiveOutput:
        """Evaluate the flow-matching loss for one microbatch.

        Args:
            model: A module honouring the ``ModelInput -> ModelOutput`` ABI.
            batch: Clean latents on a dense grid.
            rng: Purpose-separated streams. ``timestep``, ``noise``, and
                ``conditioning`` are all consumed here.
            patchifier: Converts the video grid into tokens.
            cp_mesh: The context-parallel sub-mesh, or ``None``.

        Returns:
            The loss and its reporting components.
        """
        plan = self.conditioning_sampler.sample(batch, rng=rng)
        if self.config.validate:
            plan.validate(batch)

        video_layout = patchifier.layout_for(tuple(batch.video.shape))
        audio_grid = _audio_as_grid(batch)
        has_audio = batch.spec.has_audio
        sequence_length = video_layout.num_tokens
        if has_audio:
            sequence_length += self.audio_patchifier.layout_for(
                tuple(audio_grid.shape)
            ).num_tokens

        # The shift is a function of the sequence the *model* sees, so both
        # modalities are counted and the count is post-patchification.
        video_t = self.timestep_sampler.sample(
            batch.spec.batch_size,
            device=batch.device,
            generator=rng.timestep,
            sequence_length=sequence_length,
        )
        audio_t = (
            self.timestep_sampler.sample(
                batch.spec.batch_size,
                device=batch.device,
                generator=rng.timestep,
                sequence_length=sequence_length,
            )
            if self.config.independent_modality_noise
            else video_t
        )

        video_stream, video_target = self._prepare(
            patchifier,
            grid=batch.video,
            positions=batch.video_positions,
            mask=batch.video_mask,
            conditioned=plan.video_conditioned,
            timesteps=video_t,
            rng=rng,
            cp_mesh=cp_mesh,
        )
        if has_audio:
            audio_stream, audio_target = self._prepare(
                self.audio_patchifier,
                grid=audio_grid,
                positions=batch.audio_positions,
                mask=batch.audio_mask[:, :, None, None],
                conditioned=plan.audio_conditioned[:, :, None, None],
                timesteps=audio_t,
                rng=rng,
                cp_mesh=cp_mesh,
            )
        else:
            # An absent modality still needs a well-formed stream so that every
            # audio path in the model degenerates to a no-op instead of a branch.
            # The width is the audio patch dimension the model would have seen,
            # which keeps a projection's input size stable across buckets.
            audio_stream = TokenStream.empty_like(
                batch.spec.batch_size,
                batch.spec.audio_shape[1],
                device=batch.device,
                dtype=batch.video.dtype,
            )
            audio_target = audio_stream.tokens

        inputs = ModelInput(
            video=video_stream,
            audio=audio_stream,
            text=_apply_text_dropout(batch, plan),
            condition_mode=plan.condition_mode,
        )
        if self.config.validate:
            inputs.validate()
        output: ModelOutput = model(inputs)
        if self.config.validate:
            output.validate(inputs)

        return self._reduce(
            output=output,
            video_stream=video_stream,
            video_target=video_target,
            video_t=video_t,
            audio_stream=audio_stream,
            audio_target=audio_target,
            audio_t=audio_t,
            has_audio=has_audio,
            cp_mesh=cp_mesh,
        )

    def _prepare(
        self,
        patchifier: Patchifier,
        *,
        grid: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        conditioned: torch.Tensor,
        timesteps: torch.Tensor,
        rng: RNGStreams,
        cp_mesh: DeviceMesh | None,
    ) -> tuple[TokenStream, torch.Tensor]:
        """Patchify, noise, and context-parallel shard one modality.

        Noising happens in *token* space rather than grid space. Patchification
        is a pure reshape, so the two are distributionally identical, but doing
        it after the fold means the clean-anchor mask has already been reduced
        to whole tokens by the patchifier's ``all`` rule. A patch that straddles
        the boundary of an inpainting mask is then correctly treated as noised
        and supervised, instead of receiving a fractional noise level that no
        sampler could ever reproduce at inference.

        Args:
            patchifier: Converts this modality's grid into tokens.
            grid: ``(batch, channels, frames, height, width)`` clean latents.
            positions: ``(batch, frames)`` physical time in seconds.
            mask: ``(batch, frames, height, width)`` latent validity.
            conditioned: ``(batch, frames, height, width)`` clean-anchor mask.
            timesteps: ``(batch,)`` noise levels.
            rng: Streams; ``rng.noise`` is consumed.
            cp_mesh: The context-parallel sub-mesh, or ``None``.

        Returns:
            The noisy stream held by this rank, and the matching velocity
            target in token space.
        """
        clean = patchifier.to_tokens(
            grid,
            positions=positions,
            mask=mask,
            noise_level=timesteps,
            conditioned=conditioned,
        )
        noise = _draw_noise(clean.tokens, generator=rng.noise)
        supervised = (~clean.conditioned).to(clean.tokens.dtype)
        per_token_t = clean.noise_level[:, None] * supervised
        level = per_token_t[..., None]
        noisy = clean.tokens + level * (noise - clean.tokens)
        target = noise - clean.tokens

        stream = replace(clean, tokens=noisy, noise_level=per_token_t)
        if cp_mesh is None or cp_mesh.size() == 1:
            return stream, target

        # The target must be sliced by exactly the same rule as the input, and
        # the cheapest way to guarantee that is to make it impossible to do
        # otherwise: concatenate along the *feature* axis, which context
        # parallelism never touches, shard once, then split. Sharding the two
        # separately would work today and silently diverge the day the sharding
        # rule stops being a contiguous slice.
        width = noisy.shape[-1]
        packed = shard_stream(
            replace(stream, tokens=torch.cat((noisy, target), dim=-1)), cp_mesh
        )
        sharded = replace(packed, tokens=packed.tokens[..., :width].contiguous())
        return sharded, packed.tokens[..., width:].contiguous()

    def _timestep_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch,)`` per-sample loss weight for a noise level.

        Args:
            timesteps: ``(batch,)`` noise levels in ``[0, 1]``.

        Returns:
            ``(batch,)`` non-negative float32 weights.
        """
        clamped = timesteps.to(torch.float32).clamp(_TIMESTEP_EPS, 1.0 - _TIMESTEP_EPS)
        if self.config.timestep_weighting == "sigma_sqrt":
            weight = clamped.pow(-2.0)
        elif self.config.timestep_weighting == "cosmap":
            bottom = 1.0 - 2.0 * clamped + 2.0 * clamped.pow(2)
            weight = 2.0 / (torch.pi * bottom)
        else:
            weight = torch.ones_like(clamped)
        if self.config.snr_gamma is not None:
            signal_to_noise = ((1.0 - clamped) / clamped).pow(2)
            weight = weight * (
                signal_to_noise.clamp(max=self.config.snr_gamma)
                / (signal_to_noise + 1.0)
            )
        if self.weight_fn is not None:
            weight = weight * self.weight_fn(timesteps).to(torch.float32)
        return weight

    def _reduce(
        self,
        *,
        output: ModelOutput,
        video_stream: TokenStream,
        video_target: torch.Tensor,
        video_t: torch.Tensor,
        audio_stream: TokenStream,
        audio_target: torch.Tensor,
        audio_t: torch.Tensor,
        has_audio: bool,
        cp_mesh: DeviceMesh | None,
    ) -> ObjectiveOutput:
        """Reduce per-token errors into the loss and its reporting components.

        The context-parallel handling is the part worth reading twice.

        Each rank can only see its own slice of the sequence, so it computes a
        local numerator ``N_r`` (the summed weighted squared error over its
        supervised tokens) and a local denominator ``D_r`` (its supervised token
        count). The true global loss is ``sum_r N_r / sum_r D_r``, which is
        *not* the average of ``N_r / D_r`` unless every shard has the same
        number of supervised tokens. Image-to-video guarantees they do not: the
        clean anchor is the first frames, so the first shard has systematically
        fewer supervised tokens than the rest.

        The denominator carries no gradient, so it is all-reduced directly. The
        numerator does, and it must not be all-reduced, because gradients from
        other ranks' tokens are produced on those ranks and combined by the
        parameter-gradient reduction that FSDP already performs across
        ``dp_shard_cp``. That reduction is a *mean* over the mesh, so a rank
        that backwards ``N_r / D`` contributes ``N_r / (R * D)`` and the sum
        comes out ``R`` times too small. Multiplying the local loss by the
        context-parallel degree cancels it exactly, for any shard imbalance:

        .. code-block:: text

            mean_r [ R * N_r / D ] = (1 / R) * sum_r R * N_r / D = sum_r N_r / D

        Args:
            output: The model's token-space predictions.
            video_stream: The sharded noisy video stream.
            video_target: The matching video velocity target.
            video_t: ``(batch,)`` video noise levels.
            audio_stream: The sharded noisy audio stream.
            audio_target: The matching audio velocity target.
            audio_t: ``(batch,)`` audio noise levels.
            has_audio: Whether the batch carries audio at all.
            cp_mesh: The context-parallel sub-mesh, or ``None``.

        Returns:
            The loss and its reporting components.
        """
        video_num, video_raw, video_den, video_valid = _modality_terms(
            output.video, video_target, video_stream, self._timestep_weight(video_t)
        )
        if has_audio:
            audio_num, audio_raw, audio_den, audio_valid = _modality_terms(
                output.audio, audio_target, audio_stream, self._timestep_weight(audio_t)
            )
        else:
            zero = torch.zeros((), dtype=torch.float32, device=video_num.device)
            audio_num, audio_raw, audio_den, audio_valid = zero, zero, zero, zero

        # One collective per step rather than six: the reporting numerators, the
        # loss denominators, and the valid-token counts travel together. At a
        # thousand ranks the latency of a collective dwarfs its payload.
        stats = all_reduce_sum(
            torch.stack(
                (video_raw, video_den, video_valid, audio_raw, audio_den, audio_valid)
            ),
            cp_mesh,
        )
        video_den_global = stats[1].clamp(min=1.0)
        audio_den_global = stats[4].clamp(min=1.0)
        degree = float(cp_mesh.size()) if cp_mesh is not None else 1.0

        loss = degree * (
            self.config.video_weight * video_num / video_den_global
            + self.config.audio_weight * audio_num / audio_den_global
        )
        return ObjectiveOutput(
            loss=loss.to(torch.float32),
            video_loss=(stats[0] / video_den_global).detach(),
            audio_loss=(stats[3] / audio_den_global).detach(),
            valid_video_tokens=stats[2].detach().to(torch.int64),
            valid_audio_tokens=stats[5].detach().to(torch.int64),
        )


def _modality_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stream: TokenStream,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the weighted numerator, raw numerator, denominator, and token count.

    The error is reduced in fp32 regardless of the autocast dtype. A bf16 sum
    over a hundred thousand tokens loses the low-order bits of every addend once
    the running total is a few orders of magnitude larger than the increment,
    which biases the loss downwards by an amount that grows with sequence
    length — that is, exactly with the thing being scaled up.

    Args:
        prediction: ``(batch, length, patch_dim)`` model output.
        target: ``(batch, length, patch_dim)`` velocity target.
        stream: The stream the prediction corresponds to.
        weight: ``(batch,)`` per-sample loss weight.

    Returns:
        Weighted numerator (attached), raw numerator (detached), supervised
        token count, and unpadded token count.
    """
    residual = prediction.to(torch.float32) - target.to(torch.float32)
    error = residual.pow(2).mean(dim=-1)
    supervised = stream.loss_mask().to(torch.float32)
    masked = error * supervised
    return (
        (masked * weight[:, None]).sum(),
        masked.sum().detach(),
        supervised.sum().detach(),
        stream.mask.sum().to(torch.float32).detach(),
    )


def _draw_noise(like: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Draw the Gaussian noise tensor from an explicit generator.

    This is the one draw in a training step whose size is the size of the
    activations, so it is generated directly on the compute device when the
    generator already lives there. When it does not, the tensor is drawn on the
    generator's device and copied — correct and reproducible, but a
    host-to-device transfer of activation-sized data every step. Build
    :class:`~avgen.core.rng.RNGStreams` on the compute device to avoid it; the
    alternative of falling back to the global RNG would hide the cost and break
    reproducibility, which is a far worse trade.

    Args:
        like: Tensor whose shape, dtype, and device the noise must match.
        generator: The noise stream.

    Returns:
        Standard normal noise shaped like ``like``.
    """
    if generator.device == like.device:
        return torch.randn(
            like.shape, generator=generator, device=like.device, dtype=like.dtype
        )
    drawn = torch.randn(
        like.shape, generator=generator, device=generator.device, dtype=torch.float32
    )
    return drawn.to(device=like.device, dtype=like.dtype)


def _audio_as_grid(batch: MediaBatch) -> torch.Tensor:
    """View the ``(batch, channels, frames)`` audio latent as a rank-5 grid.

    Every geometry helper in :mod:`avgen.core.patchify` is written against the
    rank-5 video shape. Rather than duplicating them for one dimension fewer,
    audio is given unit spatial extent, which the layout already understands as
    the temporal-only case.

    Args:
        batch: The batch to read.

    Returns:
        ``(batch, channels, frames, 1, 1)`` audio latents.
    """
    return batch.audio[:, :, :, None, None]


def _apply_text_dropout(batch: MediaBatch, plan: ConditioningPlan) -> TextContext:
    """Return the text context with dropped samples nulled.

    Both the features and the mask are zeroed, matching
    :func:`~avgen.core.batch.null_text_conditioning`. Zeroing only the features
    would leave a mask that says "attend to these positions", and the model
    would learn that the unconditional branch is a real prompt made entirely of
    zeros — a prompt it would then reproduce whenever guidance is applied.

    Args:
        batch: The batch supplying the conditional context.
        plan: The plan supplying the per-sample dropout decision.

    Returns:
        The possibly-nulled context.
    """
    keep = ~plan.drop_text
    features = batch.text * keep[:, None, None].to(batch.text.dtype)
    return TextContext(features=features, mask=batch.text_mask & keep[:, None])
