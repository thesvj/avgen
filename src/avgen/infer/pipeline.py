"""The generation pipeline: prompts in, media out.

The pipeline is deliberately thin. It owns no numerics: the schedule comes from
:mod:`avgen.infer.schedule`, the solver from :mod:`avgen.infer.sampler`, the
guidance from :mod:`avgen.infer.guidance`, and the model input from
:mod:`avgen.infer.conditioning` — which is the same builder the training
objective uses. What the pipeline owns is *sequencing*: encode text, allocate
noise, run the loop, decode, and record exactly what it did.

Three properties are load-bearing.

**Determinism.** Every random draw comes from
:attr:`~avgen.core.rng.RNGStreams.sampler`, seeded from the caller's seed and
from nothing else. Two runs with the same seed and the same config produce
bit-identical latents. A sample that cannot be reproduced cannot be debugged, and
"it looked better yesterday" is not a bug report anyone can act on.

**Reproducibility beyond the seed.** :class:`GeneratedMedia` carries the entire
resolved configuration — schedule, shift, sampler, guidance, shapes, codec
fingerprints, seed. A seed alone is worthless without the config that consumed
it, and the config is exactly the thing that gets edited between runs.

**Context-parallel awareness.** A long video is tens of thousands of tokens and
does not fit on one device at inference any more than it does at training. When a
``cp`` mesh is supplied, the token streams are sharded across it for the forward
pass and the predictions are gathered before the sampler step, so the solver
always sees the full sequence and its arithmetic is identical to the single-device
case.

The classifier-free-guidance branches are evaluated as two separate forward
passes rather than as one doubled batch. The doubled batch is faster — one kernel
launch sequence instead of two, and better occupancy at small batch sizes — but
it doubles peak activation memory at exactly the sequence lengths where memory is
already the binding constraint, and it forces the two branches to share a
context-parallel sharding. Two passes is the choice that scales; the doubled
batch is the right optimisation for short clips and is left to a caller who knows
its own shapes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from avgen.codecs.normalization import (
    LatentStatistics,
    denormalize_latents,
    resolve_latent_normalization,
)
from avgen.core.model_input import ModelInput, ModelOutput
from avgen.core.patchify import GridPatchifier, Patchifier, unpatchify_grid
from avgen.core.rng import RNGStreams
from avgen.core.tokens import TextContext
from avgen.infer.conditioning import (
    StreamState,
    build_model_input,
    default_audio_patchifier,
)
from avgen.infer.guidance import GuidanceConfig, apply_guidance
from avgen.infer.sampler import Sampler, SamplerConfig, build_sampler
from avgen.infer.schedule import ScheduleConfig, SigmaSchedule, build_sigma_schedule

__all__ = [
    "GeneratedMedia",
    "GenerationConfig",
    "GenerationPipeline",
]

#: ``(step_index, total_steps, sigma) -> None``.
ProgressCallback = Callable[[int, int, float], None]
#: ``(step_index, latents) -> None``, called with the normalised latent grid.
LatentCallback = Callable[[int, torch.Tensor], None]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Everything that determines one generation, pinned and serialisable.

    Args:
        steps: Number of sampler steps.
        seed: Base seed. The only source of randomness in the pipeline.
        height: Output pixel height.
        width: Output pixel width.
        num_frames: Output pixel frames.
        fps: Output frame rate. Also determines the physical coordinates the
            model sees, so changing it changes the sample, not just the metadata.
        guidance: Guidance configuration.
        sampler: Sampler configuration.
        schedule: Noise schedule configuration.
        negative_prompt: Default negative prompt used when none is supplied per
            call. ``None`` uses the zeroed null context instead, which is what
            the model was trained on and is cheaper and better behaved than
            encoding an empty string.
        generate_audio: Whether to generate an audio stream alongside the video.
        audio_seconds: Audio duration. Defaults to the video duration.
        dtype: Torch dtype name for the sampling loop.

    Raises:
        ValueError: If a size is not positive or ``fps`` is not positive.
    """

    steps: int = 30
    seed: int = 0
    height: int = 256
    width: int = 256
    num_frames: int = 17
    fps: float = 24.0
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    negative_prompt: str | None = None
    generate_audio: bool = False
    audio_seconds: float | None = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """Validate the geometry."""
        for name in ("steps", "height", "width", "num_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"seed must be non-negative; got {self.seed!r}")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be finite and positive; got {self.fps!r}")
        if self.audio_seconds is not None and (
            not math.isfinite(self.audio_seconds) or self.audio_seconds <= 0.0
        ):
            raise ValueError(
                f"audio_seconds must be finite and positive; got {self.audio_seconds!r}"
            )

    @property
    def duration_seconds(self) -> float:
        """Video duration implied by the frame count and frame rate."""
        return self.num_frames / self.fps

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the whole configuration."""
        return {
            "steps": self.steps,
            "seed": self.seed,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "guidance": self.guidance.to_dict(),
            "sampler": self.sampler.to_dict(),
            "schedule": self.schedule.to_dict(),
            "negative_prompt": self.negative_prompt,
            "generate_audio": self.generate_audio,
            "audio_seconds": self.audio_seconds,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GenerationConfig:
        """Rebuild a config from :meth:`to_dict` output.

        Args:
            values: The mapping.

        Returns:
            The restored config.
        """
        fields = dict(values)
        if "guidance" in fields:
            fields["guidance"] = GuidanceConfig.from_dict(fields["guidance"])
        if "sampler" in fields:
            fields["sampler"] = SamplerConfig.from_dict(fields["sampler"])
        if "schedule" in fields:
            fields["schedule"] = ScheduleConfig.from_dict(fields["schedule"])
        known = {
            "steps",
            "seed",
            "height",
            "width",
            "num_frames",
            "fps",
            "guidance",
            "sampler",
            "schedule",
            "negative_prompt",
            "generate_audio",
            "audio_seconds",
            "dtype",
        }
        return cls(**{key: value for key, value in fields.items() if key in known})


@dataclass(frozen=True, slots=True)
class GeneratedMedia:
    """One batch of generated media plus everything needed to reproduce it.

    The pinned config is not documentation. It is the reason a sample can be
    regenerated six months later from a filename: the seed alone reproduces
    nothing if the shift, the sampler, or the guidance schedule has moved in the
    meantime, and those are exactly the fields people edit while iterating.

    Args:
        video: ``(batch, channels, frames, height, width)`` decoded pixels.
        prompts: The prompts, one per sample.
        fps: Frame rate of ``video``.
        config: The fully resolved configuration, JSON-safe.
        audio: ``(batch, channels, samples)`` waveform, or ``None``.
        sample_rate: Sample rate of ``audio``, or ``None``.
        latents: The final normalised latents, when the caller asked for them.
        negative_prompts: The negative prompts actually used, one per sample.
    """

    video: torch.Tensor
    prompts: tuple[str, ...]
    fps: float
    config: Mapping[str, Any]
    audio: torch.Tensor | None = None
    sample_rate: int | None = None
    latents: torch.Tensor | None = None
    negative_prompts: tuple[str, ...] = ()

    @property
    def batch_size(self) -> int:
        """Number of samples."""
        return int(self.video.shape[0])

    def __len__(self) -> int:
        """Number of samples."""
        return self.batch_size

    @property
    def has_audio(self) -> bool:
        """Whether an audio stream was generated."""
        return self.audio is not None

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        return float(self.video.shape[2]) / self.fps

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-safe record of this generation.

        Returns:
            The prompts, shapes, and the pinned config — everything except the
            tensors, so it can be written next to the media file.
        """
        return {
            "prompts": list(self.prompts),
            "negative_prompts": list(self.negative_prompts),
            "video_shape": list(self.video.shape),
            "audio_shape": (list(self.audio.shape) if self.audio is not None else None),
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "config": dict(self.config),
        }


class GenerationPipeline:
    """Runs a trained denoiser end to end, from prompts to decoded media.

    Args:
        model: Any callable satisfying the model ABI —
            ``ModelInput -> ModelOutput``. Typed loosely on purpose: the pipeline
            must not import :mod:`avgen.models`, both to keep the dependency
            graph a tree and so that a test can drive it with a stub.
        video_codec: Codec used to decode latents to pixels and to derive the
            latent geometry.
        text_encoder: Text tower. ``None`` generates unconditionally.
        audio_codec: Audio codec, required when generating audio.
        patchifier: Video patchifier. Must match the model's. Defaults to the
            model's ``patchifier`` attribute when it exposes one, else to
            :class:`~avgen.core.patchify.GridPatchifier`.
        audio_patchifier: Audio patchifier.
        config: Default generation configuration; every field is overridable per
            call.
        device: Device to sample on. Defaults to the model's first parameter's
            device, falling back to CPU.
        latent_statistics: Latent normalisation. Defaults to the codec's.
        cp_mesh: Context-parallel sub-mesh, or ``None``.

    Raises:
        ValueError: If audio generation is configured without an audio codec.
    """

    __slots__ = (
        "audio_codec",
        "audio_patchifier",
        "config",
        "cp_mesh",
        "device",
        "latent_statistics",
        "model",
        "patchifier",
        "text_encoder",
        "video_codec",
    )

    def __init__(
        self,
        model: Any,
        *,
        video_codec: Any,
        text_encoder: Any = None,
        audio_codec: Any = None,
        patchifier: Patchifier | None = None,
        audio_patchifier: Patchifier | None = None,
        config: GenerationConfig | None = None,
        device: torch.device | str | None = None,
        latent_statistics: LatentStatistics | None = None,
        cp_mesh: Any = None,
    ) -> None:
        self.model = model
        self.video_codec = video_codec
        self.text_encoder = text_encoder
        self.audio_codec = audio_codec
        self.config = config or GenerationConfig()
        if self.config.generate_audio and audio_codec is None:
            raise ValueError(
                "generate_audio is set but no audio_codec was supplied; the "
                "pipeline cannot decode an audio latent without one"
            )
        self.patchifier = patchifier or _model_patchifier(model)
        self.audio_patchifier = audio_patchifier or default_audio_patchifier()
        self.device = (
            torch.device(device) if device is not None else _model_device(model)
        )
        self.latent_statistics = latent_statistics or resolve_latent_normalization(
            video_codec, channels=getattr(video_codec, "latent_channels", None)
        )
        self.cp_mesh = cp_mesh

    def __call__(
        self,
        prompts: Sequence[str] | str,
        *,
        steps: int | None = None,
        guidance: GuidanceConfig | float | None = None,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
        negative_prompts: Sequence[str] | str | None = None,
        sampler: SamplerConfig | str | None = None,
        schedule: ScheduleConfig | None = None,
        generate_audio: bool | None = None,
        progress_callback: ProgressCallback | None = None,
        latent_callback: LatentCallback | None = None,
        return_latents: bool = False,
    ) -> GeneratedMedia:
        """Generate media for a batch of prompts.

        Args:
            prompts: One prompt, or a batch of them.
            steps: Sampler steps, overriding the config.
            guidance: Guidance config, or a bare float used as the scale.
            seed: Base seed, overriding the config.
            height: Output pixel height.
            width: Output pixel width.
            num_frames: Output pixel frames.
            fps: Output frame rate.
            negative_prompts: Negative prompt or prompts. When given, the
                guidance null branch is the *encoding of these* rather than the
                zeroed context — the model is steered away from them rather than
                merely away from nothing.
            sampler: Sampler config or name.
            schedule: Schedule config.
            generate_audio: Whether to produce audio.
            progress_callback: Called after each step with
                ``(index, total, sigma)``.
            latent_callback: Called after each step with ``(index, latents)``,
                where ``latents`` are the current normalised latents. Useful for
                previews; note it forces the latents to stay materialised.
            return_latents: Whether to return the final latents alongside the
                decoded media.

        Returns:
            The generated media and the configuration that produced it.

        Raises:
            ValueError: If audio was requested without an audio codec, or the
                negative prompt count does not match the batch.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompt_list:
            raise ValueError("prompts must contain at least one entry")
        config = self._resolve(
            steps=steps,
            guidance=guidance,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            sampler=sampler,
            schedule=schedule,
            generate_audio=generate_audio,
            negative_prompts=negative_prompts,
        )
        negatives = _resolve_negatives(negative_prompts, config, len(prompt_list))
        if config.generate_audio and self.audio_codec is None:
            raise ValueError(
                "audio generation was requested but the pipeline has no audio_codec"
            )

        dtype = getattr(torch, config.dtype)
        rng = RNGStreams.from_seed(config.seed, device=self.device)
        text = self._encode(prompt_list, dtype=dtype)
        null_text = (
            self._encode(negatives, dtype=dtype) if any(negatives) else text.nullified()
        )

        video_state, audio_state = self._initial_states(
            batch=len(prompt_list), config=config, rng=rng, dtype=dtype
        )
        sequence_length = _sequence_length(
            video_state, audio_state, self.patchifier, self.audio_patchifier
        )
        sigma_schedule = build_sigma_schedule(
            config.schedule, sequence_length=sequence_length, device=self.device
        )
        solver = build_sampler(config.sampler)
        solver.reset()

        video_state, audio_state = self._sample(
            video_state=video_state,
            audio_state=audio_state,
            text=text,
            null_text=null_text,
            schedule=sigma_schedule,
            solver=solver,
            config=config,
            rng=rng,
            progress_callback=progress_callback,
            latent_callback=latent_callback,
        )

        video = self._decode_video(video_state.latents)
        audio = (
            self._decode_audio(audio_state.latents) if audio_state is not None else None
        )
        pinned = dict(config.to_dict())
        pinned["resolved_shift"] = sigma_schedule.shift
        pinned["sequence_length"] = sequence_length
        pinned["video_codec_id"] = getattr(self.video_codec, "fingerprint", None)
        pinned["audio_codec_id"] = getattr(self.audio_codec, "fingerprint", None)
        pinned["text_encoder_id"] = getattr(self.text_encoder, "fingerprint", None)
        pinned["latent_statistics"] = self.latent_statistics.to_dict()
        return GeneratedMedia(
            video=video,
            audio=audio,
            fps=config.fps,
            sample_rate=(
                int(getattr(self.audio_codec, "sample_rate", 0)) or None
                if audio is not None
                else None
            ),
            prompts=tuple(prompt_list),
            negative_prompts=tuple(negatives),
            config=pinned,
            latents=video_state.latents if return_latents else None,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        video_codec: Any,
        text_encoder: Any = None,
        audio_codec: Any = None,
        config: GenerationConfig | None = None,
        device: torch.device | str | None = None,
        model_name: str | None = None,
        model_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationPipeline:
        """Build a pipeline from a checkpoint directory or safetensors file.

        The model class and its config are read from a ``config.json`` sitting
        next to the weights, unless ``model_name`` is given explicitly. The
        generation defaults are read from the same file's ``generation`` key when
        present, so a checkpoint can ship the shift it was trained with — which
        is the field most likely to be got wrong (see
        :mod:`avgen.infer.schedule`).

        Both :mod:`avgen.models` and :mod:`avgen.checkpoint` are imported inside
        this method, so importing :mod:`avgen.infer` never pulls them in.

        Args:
            path: Checkpoint directory, or a ``.safetensors`` file.
            video_codec: Codec to decode with.
            text_encoder: Text tower.
            audio_codec: Audio codec.
            config: Generation defaults, overriding anything the checkpoint
                declares.
            device: Device to sample on.
            model_name: Registered model name, overriding the checkpoint's.
            model_config: Model configuration, overriding the checkpoint's.
            **kwargs: Forwarded to the constructor.

        Returns:
            A ready pipeline.

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If neither a safetensors file nor a usable
                :mod:`avgen.checkpoint` loader is available.
        """
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {root}")
        manifest = _read_manifest(root)
        name = model_name or manifest.get("model", {}).get("name")
        if name is None:
            raise RuntimeError(
                f"no model name found in {root}; pass model_name= explicitly or "
                "add a config.json with a 'model.name' entry"
            )
        from avgen.models import build_model

        settings = dict(model_config or manifest.get("model", {}).get("config", {}))
        model = build_model(name, settings)
        _load_weights(root, model)
        model.eval()
        target = torch.device(device) if device is not None else _model_device(model)
        model.to(target)

        defaults = config
        if defaults is None and "generation" in manifest:
            defaults = GenerationConfig.from_dict(manifest["generation"])
        return cls(
            model,
            video_codec=video_codec,
            text_encoder=text_encoder,
            audio_codec=audio_codec,
            config=defaults,
            device=target,
            **kwargs,
        )

    def _resolve(
        self,
        *,
        steps: int | None,
        guidance: GuidanceConfig | float | None,
        seed: int | None,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        fps: float | None,
        sampler: SamplerConfig | str | None,
        schedule: ScheduleConfig | None,
        generate_audio: bool | None,
        negative_prompts: Sequence[str] | str | None,
    ) -> GenerationConfig:
        """Fold the per-call overrides onto the pipeline defaults."""
        overrides: dict[str, Any] = {}
        if steps is not None:
            overrides["steps"] = steps
        if seed is not None:
            overrides["seed"] = seed
        if height is not None:
            overrides["height"] = height
        if width is not None:
            overrides["width"] = width
        if num_frames is not None:
            overrides["num_frames"] = num_frames
        if fps is not None:
            overrides["fps"] = fps
        if generate_audio is not None:
            overrides["generate_audio"] = generate_audio
        if guidance is not None:
            overrides["guidance"] = (
                GuidanceConfig(scale=float(guidance))
                if isinstance(guidance, (int, float))
                else guidance
            )
        if sampler is not None:
            overrides["sampler"] = (
                SamplerConfig(name=sampler) if isinstance(sampler, str) else sampler
            )
        if schedule is not None:
            overrides["schedule"] = schedule
        if isinstance(negative_prompts, str):
            overrides["negative_prompt"] = negative_prompts
        resolved = replace(self.config, **overrides)
        # The step count lives in two places — the top-level convenience field
        # and the schedule — and the schedule is what actually builds the sigmas.
        # Keeping them in sync here means `steps=8` does what the caller meant.
        if resolved.schedule.steps != resolved.steps:
            resolved = replace(
                resolved, schedule=replace(resolved.schedule, steps=resolved.steps)
            )
        return resolved

    def _encode(self, prompts: Sequence[str], *, dtype: torch.dtype) -> TextContext:
        """Encode prompts into a text context, or return an empty one."""
        if self.text_encoder is None:
            return TextContext.empty(len(prompts), 1, device=self.device, dtype=dtype)
        features, mask = self.text_encoder.encode(list(prompts))
        context = TextContext(
            features=features.to(device=self.device, dtype=dtype),
            mask=mask.to(device=self.device),
        )
        context.validate()
        return context

    def _initial_states(
        self,
        *,
        batch: int,
        config: GenerationConfig,
        rng: RNGStreams,
        dtype: torch.dtype,
    ) -> tuple[StreamState, StreamState | None]:
        """Allocate the pure-noise starting point for each modality."""
        pixel_shape = (
            batch,
            _codec_channels(self.video_codec),
            config.num_frames,
            config.height,
            config.width,
        )
        latent_shape = tuple(self.video_codec.latent_shape(pixel_shape))
        video_latents = torch.randn(
            latent_shape, generator=rng.sampler, device=self.device, dtype=dtype
        )
        # Latent fps, not pixel fps: the coordinates the model was trained on are
        # those of the *latent* grid, and a VAE that compresses time by four
        # means the latent frame rate is a quarter of the video's.
        latent_fps = config.fps / max(int(self.video_codec.temporal_compression), 1)
        video = StreamState(latents=video_latents, fps=latent_fps)

        if not config.generate_audio:
            return video, None
        seconds = config.audio_seconds or config.duration_seconds
        hop = int(getattr(self.audio_codec, "hop_length", 256))
        rate = int(getattr(self.audio_codec, "sample_rate", 24000))
        frames = max(round(seconds * rate / hop), 1)
        channels = int(getattr(self.audio_codec, "latent_channels", 1))
        audio_latents = torch.randn(
            (batch, channels, frames),
            generator=rng.sampler,
            device=self.device,
            dtype=dtype,
        )
        audio = StreamState(latents=audio_latents, fps=rate / hop)
        return video, audio

    def _sample(
        self,
        *,
        video_state: StreamState,
        audio_state: StreamState | None,
        text: TextContext,
        null_text: TextContext,
        schedule: SigmaSchedule,
        solver: Sampler,
        config: GenerationConfig,
        rng: RNGStreams,
        progress_callback: ProgressCallback | None,
        latent_callback: LatentCallback | None,
    ) -> tuple[StreamState, StreamState | None]:
        """Run the sampling loop.

        The video and audio streams get *separate* solver instances. A multistep
        solver carries the previous step's clean-sample prediction, and feeding
        one instance alternating video and audio predictions would extrapolate
        each modality from the other's history — a bug that produces plausible
        but subtly wrong output and no error at all.
        """
        total = schedule.num_steps
        audio_solver: Sampler | None = None
        if audio_state is not None:
            audio_solver = build_sampler(config.sampler)
            audio_solver.reset()
        with torch.no_grad():
            for index, sigma, sigma_next in schedule:
                video_state, audio_state = self._advance(
                    video_state=video_state,
                    audio_state=audio_state,
                    text=text,
                    null_text=null_text,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    progress=index / max(total - 1, 1),
                    solver=solver,
                    audio_solver=audio_solver,
                    config=config,
                    rng=rng,
                )
                if progress_callback is not None:
                    progress_callback(index, total, sigma_next)
                if latent_callback is not None:
                    latent_callback(index, video_state.latents)
        return video_state, audio_state

    def _advance(
        self,
        *,
        video_state: StreamState,
        audio_state: StreamState | None,
        text: TextContext,
        null_text: TextContext,
        sigma: float,
        sigma_next: float,
        progress: float,
        solver: Sampler,
        audio_solver: Sampler | None,
        config: GenerationConfig,
        rng: RNGStreams,
    ) -> tuple[StreamState, StreamState | None]:
        """Advance every stream by one sampler step.

        Kept as a method taking explicit arguments rather than a closure inside
        the loop: the two-evaluation solvers need a callback that re-enters the
        model, and a callback closing over a loop variable is the classic way to
        capture the wrong iteration's state.

        Args:
            video_state: Current video stream state.
            audio_state: Current audio stream state, or ``None``.
            text: Conditional text context.
            null_text: Null or negative text context.
            sigma: Current flow time.
            sigma_next: Target flow time.
            progress: Trajectory position for the guidance schedule.
            solver: Video solver.
            audio_solver: Audio solver, or ``None``.
            config: Resolved generation config.
            rng: Random streams.

        Returns:
            The advanced states.
        """
        audio_latents = audio_state.latents if audio_state is not None else None

        def evaluate(
            video_latents: torch.Tensor,
            current_audio: torch.Tensor | None,
            at: float,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            return self._velocity(
                video_state=video_state.with_latents(video_latents),
                audio_state=(
                    audio_state.with_latents(current_audio)
                    if audio_state is not None and current_audio is not None
                    else None
                ),
                text=text,
                null_text=null_text,
                sigma=at,
                config=config,
                progress=progress,
            )

        def denoise_video(probe: torch.Tensor, at: float) -> torch.Tensor:
            velocity, _ = evaluate(probe, audio_latents, at)
            return velocity

        video_velocity, audio_velocity = evaluate(
            video_state.latents, audio_latents, sigma
        )
        next_video = video_state.with_latents(
            solver.step(
                video_velocity,
                video_state.latents,
                sigma,
                sigma_next,
                denoise=denoise_video,
                generator=rng.sampler,
            )
        )
        next_audio = audio_state
        if (
            audio_state is not None
            and audio_velocity is not None
            and audio_solver is not None
        ):
            next_audio = audio_state.with_latents(
                audio_solver.step(
                    audio_velocity,
                    audio_state.latents,
                    sigma,
                    sigma_next,
                    denoise=None,
                    generator=rng.sampler,
                )
            )
        return next_video, next_audio

    def _velocity(
        self,
        *,
        video_state: StreamState,
        audio_state: StreamState | None,
        text: TextContext,
        null_text: TextContext,
        sigma: float,
        config: GenerationConfig,
        progress: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evaluate the model, apply guidance, and unfold to latent grids."""
        inputs = build_model_input(
            video=video_state,
            sigma=sigma,
            text=text,
            patchifier=self.patchifier,
            audio=audio_state,
            audio_patchifier=self.audio_patchifier,
        )
        conditional = self._forward(inputs)
        unconditional: ModelOutput | None = None
        if config.guidance.is_enabled:
            null_inputs = replace(inputs, text=null_text)
            unconditional = self._forward(null_inputs)

        video = apply_guidance(
            conditional.video,
            unconditional.video if unconditional is not None else None,
            config=config.guidance,
            progress=progress,
            modality="text",
        )
        video_grid = unpatchify_grid(video, inputs.video.layout)

        audio_grid: torch.Tensor | None = None
        if audio_state is not None and inputs.has_audio:
            audio = apply_guidance(
                conditional.audio,
                unconditional.audio if unconditional is not None else None,
                config=config.guidance,
                progress=progress,
                modality="audio",
            )
            audio_grid = unpatchify_grid(audio, inputs.audio.layout)[..., 0, 0]
        return video_grid, audio_grid

    def _forward(self, inputs: ModelInput) -> ModelOutput:
        """Run one forward pass, sharding and gathering when context-parallel.

        Under context parallelism every rank holds a contiguous slice of the
        token sequence for the forward pass, and the predictions are gathered
        before returning, so the sampler and the guidance arithmetic always see
        the full sequence. Keeping the gather here rather than in the sampler is
        what lets every solver stay a pure function of full-length tensors.
        """
        if self.cp_mesh is None:
            return self.model(inputs)

        from avgen.parallel.context import gather_tokens, shard_stream

        sharded = inputs.replace_streams(
            video=shard_stream(inputs.video, self.cp_mesh),
            audio=(
                shard_stream(inputs.audio, self.cp_mesh)
                if inputs.has_audio
                else inputs.audio
            ),
        )
        output = self.model(sharded)
        return ModelOutput(
            video=gather_tokens(output.video, self.cp_mesh),
            audio=(
                gather_tokens(output.audio, self.cp_mesh)
                if inputs.has_audio
                else output.audio
            ),
            auxiliary=tuple(
                gather_tokens(value, self.cp_mesh) for value in output.auxiliary
            ),
        )

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalise and decode video latents to pixels."""
        return self.video_codec.decode(
            denormalize_latents(latents, self.latent_statistics)
        )

    def _decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalise and decode audio latents to a waveform."""
        statistics = resolve_latent_normalization(
            self.audio_codec, channels=int(latents.shape[1])
        )
        return self.audio_codec.decode(denormalize_latents(latents, statistics))


def _sequence_length(
    video: StreamState,
    audio: StreamState | None,
    patchifier: Patchifier,
    audio_patchifier: Patchifier,
) -> int:
    """Return the global token count, which the dynamic shift is a function of.

    Global, not per-rank: a context-parallel run must resolve the same shift as
    a single-device run of the same sample, or the two produce different videos
    from the same seed.
    """
    total = patchifier.layout_for(tuple(video.grid().shape)).num_tokens
    if audio is not None:
        total += audio_patchifier.layout_for(tuple(audio.grid().shape)).num_tokens
    return total


def _resolve_negatives(
    negative_prompts: Sequence[str] | str | None,
    config: GenerationConfig,
    batch: int,
) -> list[str]:
    """Broadcast the negative prompt to one entry per sample."""
    if negative_prompts is None:
        default = config.negative_prompt or ""
        return [default] * batch
    if isinstance(negative_prompts, str):
        return [negative_prompts] * batch
    values = list(negative_prompts)
    if len(values) == 1:
        return values * batch
    if len(values) != batch:
        raise ValueError(
            f"negative_prompts must have 1 or {batch} entries; got {len(values)}"
        )
    return values


def _model_patchifier(model: Any) -> Patchifier:
    """Return the model's patchifier, or the default.

    A model that patchifies differently from the pipeline produces token streams
    its weights were not trained on, so the model's own choice always wins when
    it declares one.
    """
    candidate = getattr(model, "patchifier", None)
    if candidate is not None:
        return candidate  # type: ignore[no-any-return]
    return GridPatchifier()


def _model_device(model: Any) -> torch.device:
    """Return the device the model's parameters live on, defaulting to CPU."""
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            return torch.device(parameter.device)
    return torch.device("cpu")


def _codec_channels(codec: Any) -> int:
    """Return a codec's pixel channel count, defaulting to RGB."""
    return int(getattr(codec, "channels", 3))


def _read_manifest(root: Path) -> dict[str, Any]:
    """Read the JSON manifest that sits next to a checkpoint's weights."""
    candidates = (
        root / "config.json" if root.is_dir() else root.with_suffix(".json"),
        root.parent / "config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            loaded: dict[str, Any] = json.loads(candidate.read_text())
            return loaded
    return {}


def _safetensors_files(root: Path) -> list[Path]:
    """Collect the safetensors shards under a checkpoint root.

    ``avgen.checkpoint.export_safetensors`` writes a *directory* of shards plus
    an index, and that directory is itself commonly named ``model.safetensors``.
    A plain suffix glob therefore matches the directory as well as its contents,
    so files are filtered explicitly and the search descends one level when the
    top level holds none.

    Args:
        root: Checkpoint directory or file.

    Returns:
        The shard files, in a stable order.
    """
    if root.is_file():
        return [root] if root.suffix == ".safetensors" else []
    top = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    if top:
        return top
    return sorted(path for path in root.glob("*/*.safetensors") if path.is_file())


def _load_weights(root: Path, model: Any) -> None:
    """Load weights into a freshly built model.

    Tries the release format first (the safetensors shards that
    :func:`avgen.checkpoint.export_safetensors` writes) and falls back to
    :mod:`avgen.checkpoint` for a sharded distributed checkpoint.

    Args:
        root: Checkpoint directory or file.
        model: The model to load into.

    Raises:
        RuntimeError: If no supported weight file is found, or the shards do not
            cover the model's parameters.
    """
    files = _safetensors_files(root)
    if files:
        from safetensors.torch import load_file

        state: dict[str, torch.Tensor] = {}
        for file in files:
            state.update(load_file(str(file)))
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", ()) or ())
        if missing:
            # Loading non-strictly is deliberate — an export may legitimately
            # omit tied or buffer entries — but silently leaving half a model
            # randomly initialised produces samples that look like a bad
            # checkpoint rather than like a loading bug, so say so.
            raise RuntimeError(
                f"{len(missing)} parameters were not found in the weights under "
                f"{root}; first missing: {missing[:5]}. The checkpoint does not "
                "match the model configuration it was built with."
            )
        return

    import avgen.checkpoint as checkpoint_module

    # avgen.checkpoint's documented entry points are trainer-shaped: they take a
    # TrainState and a ParallelModel. An inference-only loader is not part of the
    # frozen contract, so probe for one and give a precise error rather than
    # silently producing an untrained model.
    for name in ("load_model", "load_for_inference", "load_weights"):
        loader = getattr(checkpoint_module, name, None)
        if callable(loader):
            loader(root, model)
            return
    raise RuntimeError(
        f"no weights found under {root}: expected a .safetensors file written by "
        "avgen.checkpoint.export_safetensors, or an inference loader "
        "(load_model / load_for_inference / load_weights) on avgen.checkpoint"
    )
