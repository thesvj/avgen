"""Video and audio-video diffusion transformers.

:class:`VideoDiT` is the reference architecture: a pre-norm transformer over a
patchified latent sequence, conditioned on a noise level through adaLN-single
and on text through cross-attention, predicting a per-token field (a flow
velocity, under avgen's default objective) in token space.

:class:`AVDiT` extends it with a second generative stream and exactly two fusion
mechanisms. Two, not a family: every additional fusion variant is a
configuration that must be trained, evaluated, and supported forever, and the
two here span the useful trade-off.

* ``cross_attention`` keeps two towers with independent widths and exchanges
  information periodically through temporally-local bidirectional
  cross-attention. Audio wants far fewer channels than video — a spectrogram
  latent is not a 720p frame — and this is the mode that lets you spend
  parameters where they matter. Locality comes from the physical time
  coordinates, so "within 250 ms" means the same thing whatever the two streams'
  frame rates are.
* ``joint`` concatenates the two streams into one sequence and runs one
  self-attention over both, with a shared temporal rotary phase space. Every
  parameter is shared, cross-modal attention is exact rather than windowed, and
  there is nothing to schedule. It costs attention quadratic in the *combined*
  length, and it forces both streams through one width.

**Everything is built to be constructible on the meta device.** ``__init__``
allocates module structure and nothing else; :meth:`VideoDiT.init_weights` does
every actual write. That separation is not stylistic — a 14B model cannot be
built on one rank's memory and then sharded, so the sequence is: build under
``torch.device("meta")``, apply FSDP/TP, ``to_empty()`` onto the real device,
then ``init_weights()``. Any constant computed in ``__init__`` and stored in a
buffer is silently replaced with uninitialised memory by that ``to_empty()``,
which is why the rotary tables are computed from coordinates at call time
instead of being cached.

**Sequence sharding belongs to context parallelism, not to tensor
parallelism.** Sequence-parallel TP shards only the norm and residual regions
and all-gathers before every projection, so the model still sees the full
sequence inside attention and the rotary tables stay consistent. Context
parallelism shards the :class:`~avgen.core.tokens.TokenStream` itself —
coordinates, masks, and per-token noise levels together — which is what keeps
positions correct with no rank-aware code anywhere in this file. One consequence
is worth stating plainly: **per-token noise levels are incompatible with
sequence-parallel tensor parallelism**, because the modulation tensor is built
at the root at full sequence length while the block interior is sharded. Shard
the sequence with ``context=`` and the problem does not arise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from avgen.core.model_input import ModelInput, ModelOutput
from avgen.core.patchify import GridPatchifier, Patchifier
from avgen.core.tokens import TextContext, TokenStream
from avgen.models.attention import key_padding_mask
from avgen.models.blocks import (
    CrossModalFusion,
    DiTBlock,
    TextRefinerBlock,
    temporal_neighbour_mask,
)
from avgen.models.layers import (
    AdaLNModulation,
    TimestepEmbedding,
    init_linear,
    init_norm,
    rms_norm,
)
from avgen.models.registry import register_model
from avgen.models.rope import RoPEScaling, RotaryEmbedding, RotaryTables
from avgen.simulate.memory import ModelShape

if TYPE_CHECKING:
    from avgen.parallel.tensor import TensorParallelPlan

__all__ = [
    "FUSION_MODES",
    "AVDiT",
    "AVDiTConfig",
    "VideoDiT",
    "VideoDiTConfig",
    "list_presets",
    "preset",
]

#: The audio-video fusion mechanisms :class:`AVDiT` supports. Deliberately two.
FUSION_MODES: tuple[str, ...] = ("cross_attention", "joint")


@dataclass(frozen=True, slots=True)
class VideoDiTConfig:
    """Architecture of a video diffusion transformer.

    Args:
        width: Residual stream width.
        depth: Number of transformer blocks.
        num_heads: Attention heads. ``width`` must divide by it.
        num_kv_heads: Key/value heads for grouped-query attention. ``None``
            means multi-head attention.
        mlp_ratio: Feed-forward inner width as a multiple of ``width``. Ignored
            when ``ffn_hidden`` is set.
        ffn_hidden: Explicit feed-forward inner width.
        ffn_multiple_of: Rounding applied to the derived inner width. Matmul
            efficiency falls off a cliff when a dimension is not a multiple of
            the tensor-core tile, and a "nice" ratio frequently is not.
        in_channels: Latent channels the codec produces.
        out_channels: Predicted channels. Defaults to ``in_channels``; a model
            predicting both a velocity and a variance would double it.
        patch_frames: Temporal patch size.
        patch_height: Spatial patch height.
        patch_width: Spatial patch width.
        normalize_space: Whether the patchifier normalises spatial coordinates.
        text_width: Width of the frozen text encoder's hidden states.
        text_context_length: Declared maximum prompt length. Used only for
            memory accounting in :meth:`VideoDiT.model_shape`.
        text_refiner_depth: Bidirectional blocks applied to the text features
            before cross-attention. Zero disables the refiner.
        cross_attention: Whether blocks carry a text cross-attention branch.
        rope_theta: Rotary base.
        rope_time_scale: Multiplier on the seconds axis of the rotary phase.
        rope_scaling: Frequency scaling for length or resolution extrapolation.
        cond_width: Width of the timestep conditioning vector. Defaults to
            ``width``.
        time_frequency_dim: Width of the sinusoidal noise-level bank.
        qk_norm: Whether to RMS-normalise queries and keys per head.
        norm_eps: Normalisation epsilon.
        assume_dense_mask: Skip the all-valid mask check. Set true when the
            loader guarantees no padding; it removes one host-device sync per
            stream per forward.
        init_std: Base standard deviation for weight initialisation.

    Raises:
        ValueError: On any invalid or mutually inconsistent field.
    """

    width: int = 768
    depth: int = 24
    num_heads: int = 12
    num_kv_heads: int | None = None
    mlp_ratio: float = 4.0
    ffn_hidden: int | None = None
    ffn_multiple_of: int = 256
    in_channels: int = 16
    out_channels: int | None = None
    patch_frames: int = 1
    patch_height: int = 2
    patch_width: int = 2
    normalize_space: bool = False
    text_width: int = 4096
    text_context_length: int = 0
    text_refiner_depth: int = 0
    cross_attention: bool = True
    rope_theta: float = 10000.0
    rope_time_scale: float = 1.0
    rope_scaling: RoPEScaling = field(default_factory=RoPEScaling)
    cond_width: int | None = None
    time_frequency_dim: int = 256
    qk_norm: bool = True
    norm_eps: float = 1e-6
    assume_dense_mask: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Validate the architecture.

        Raises:
            ValueError: Naming the offending field and its value.
        """
        for name in (
            "width",
            "depth",
            "num_heads",
            "ffn_multiple_of",
            "in_channels",
            "patch_frames",
            "patch_height",
            "patch_width",
            "text_width",
            "time_frequency_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.width % self.num_heads != 0:
            raise ValueError(
                f"width {self.width} must be divisible by num_heads {self.num_heads}"
            )
        if self.head_dim % 2 != 0 or self.head_dim < 6:
            raise ValueError(
                f"head_dim {self.head_dim} (width // num_heads) must be even and at "
                "least 6 so the rotary embedding can give every axis a channel pair"
            )
        kv_heads = self.num_kv_heads
        if kv_heads is not None and (kv_heads < 1 or self.num_heads % kv_heads != 0):
            raise ValueError(
                f"num_kv_heads must divide num_heads {self.num_heads}; got {kv_heads!r}"
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive; got {self.mlp_ratio!r}")
        if self.ffn_hidden is not None and self.ffn_hidden < 1:
            raise ValueError(f"ffn_hidden must be positive; got {self.ffn_hidden!r}")
        if self.text_refiner_depth < 0:
            raise ValueError(
                "text_refiner_depth must be non-negative; got "
                f"{self.text_refiner_depth!r}"
            )
        if self.text_refiner_depth and not self.cross_attention:
            raise ValueError(
                "text_refiner_depth > 0 is meaningless without cross_attention: the "
                "refined text would never be read"
            )
        if not isinstance(self.rope_scaling, RoPEScaling):
            raise TypeError(
                f"rope_scaling must be a RoPEScaling; got {self.rope_scaling!r}"
            )

    @property
    def head_dim(self) -> int:
        """Channels per attention head."""
        return self.width // self.num_heads

    @property
    def ffn_width(self) -> int:
        """Feed-forward inner width, after rounding."""
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        hidden = int(self.mlp_ratio * self.width)
        multiple = self.ffn_multiple_of
        return -(-hidden // multiple) * multiple

    @property
    def cond_dim(self) -> int:
        """Width of the timestep conditioning vector."""
        return self.cond_width if self.cond_width is not None else self.width

    @property
    def patch_dim(self) -> int:
        """Input feature width of one packed video patch."""
        return (
            self.in_channels * self.patch_frames * self.patch_height * self.patch_width
        )

    @property
    def out_patch_dim(self) -> int:
        """Predicted feature width of one packed video patch."""
        channels = (
            self.out_channels if self.out_channels is not None else self.in_channels
        )
        return channels * self.patch_frames * self.patch_height * self.patch_width


@register_model("video_dit")
class VideoDiT(nn.Module):
    """Text-conditioned video diffusion transformer over a token sequence.

    The module graph is fixed by the tensor-parallel contract in
    :mod:`avgen.parallel.tensor`: the root exposes ``patch_embed``,
    ``time_embed``, ``text_proj``, ``blocks``, ``final_norm``, and
    ``final_proj``, and ``blocks`` is an :class:`torch.nn.ModuleList` so that
    FSDP, activation checkpointing, and ``torch.compile`` can wrap one block at
    a time.

    There is deliberately **no adaLN modulation between ``final_norm`` and
    ``final_proj``**. The frozen root plan shards those two as adjacent
    sequence-parallel and column-parallel modules, and an elementwise modulation
    between them would mix a replicated tensor with a sequence-sharded one. The
    output head starts as the identity anyway, because ``final_proj`` is
    zero-initialised.

    Args:
        config: The architecture.
    """

    def __init__(self, config: VideoDiTConfig) -> None:
        super().__init__()
        self.config = config
        self._patchifier = GridPatchifier(
            patch_frames=config.patch_frames,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
            normalize_space=config.normalize_space,
        )
        self.rope = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            time_scale=config.rope_time_scale,
            scaling=config.rope_scaling,
        )
        self.patch_embed = nn.Linear(config.patch_dim, config.width, bias=True)
        self.time_embed = TimestepEmbedding(
            config.cond_dim, frequency_dim=config.time_frequency_dim
        )
        self.text_proj = nn.Linear(config.text_width, config.width, bias=True)
        self.text_refiner = nn.ModuleList(
            TextRefinerBlock(
                config.width,
                config.num_heads,
                hidden=config.ffn_width,
                qk_norm=config.qk_norm,
                eps=config.norm_eps,
            )
            for _ in range(config.text_refiner_depth)
        )
        self.modulation = AdaLNModulation(config.cond_dim, config.width)
        self.blocks = nn.ModuleList(
            DiTBlock(
                config.width,
                config.num_heads,
                hidden=config.ffn_width,
                num_kv_heads=config.num_kv_heads,
                cross_attention=config.cross_attention,
                qk_norm=config.qk_norm,
                eps=config.norm_eps,
            )
            for _ in range(config.depth)
        )
        self.final_norm = rms_norm(config.width, eps=config.norm_eps)
        self.final_proj = nn.Linear(config.width, config.out_patch_dim, bias=True)

    @property
    def patchifier(self) -> Patchifier:
        """The patchifier whose geometry this model was built for.

        Returns:
            The patchifier the objective and sampler must use, so that the
            token width the model consumes and the grid the codec produces stay
            in agreement.
        """
        return self._patchifier

    def parameter_count(self) -> int:
        """Count parameters, unsharded.

        Works on a meta-device model, which is the point: the size of a 14B
        configuration should be knowable before anything is allocated.

        Returns:
            Total parameters.
        """
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def model_shape(
        self,
        *,
        sequence_length: int,
        micro_batch_size: int = 1,
    ) -> ModelShape:
        """Describe this model to the memory and compute simulators.

        Args:
            sequence_length: Tokens per sample before context-parallel
                sharding.
            micro_batch_size: Samples per rank per microbatch.

        Returns:
            The shape :func:`~avgen.simulate.memory.estimate_memory` prices.
        """
        config = self.config
        return ModelShape(
            parameters=self.parameter_count(),
            depth=config.depth,
            width=config.width,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            # The simulator's activation model reads mlp_ratio as an integer
            # multiplier on width, so report the *effective* ratio after
            # rounding rather than the configured one.
            mlp_ratio=max(1, round(config.ffn_width / config.width)),
            num_heads=config.num_heads,
            text_tokens=config.text_context_length if config.cross_attention else 0,
        )

    def init_weights(self) -> None:
        """Initialise every parameter in place.

        Kept out of ``__init__`` so a model can be built on the meta device,
        sharded, materialised with ``to_empty()``, and only then filled — the
        only sequence in which a model too large for one device can be created
        at all. Determinism comes from the ambient torch seed, which the
        training entry point sets per data rank.
        """
        config = self.config
        init_linear(self.patch_embed, std=config.init_std)
        self.time_embed.init_weights()
        init_linear(self.text_proj, std=config.init_std)
        for refiner in self.text_refiner:
            refiner.init_weights(
                depth=max(1, config.text_refiner_depth), std=config.init_std
            )
        self.modulation.init_weights()
        for block in self.blocks:
            block.init_weights(depth=config.depth, std=config.init_std)
        init_norm(self.final_norm)
        # Zero output head: the model predicts exactly zero at step zero, so the
        # first gradient is the target field itself rather than the difference
        # between the target and an arbitrary random projection.
        init_linear(self.final_proj, zero=True)

    def encode_text(
        self, text: TextContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Project and optionally refine the frozen text context.

        Args:
            text: The frozen encoder output.

        Returns:
            ``(features, mask)`` ready for cross-attention, or ``(None, None)``
            when there is no text conditioning and the branch should be skipped
            entirely.
        """
        if text.is_empty or not self.config.cross_attention:
            return None, None
        features = self.text_proj(text.features)
        mask = key_padding_mask(text.mask, assume_dense=self.config.assume_dense_mask)
        for refiner in self.text_refiner:
            features = refiner(features, key_mask=mask)
        return features, mask

    def _stream_conditioning(
        self,
        stream: TokenStream,
        *,
        embed: TimestepEmbedding,
        modulation: AdaLNModulation,
    ) -> torch.Tensor:
        """Build the adaLN modulation tensor for one stream."""
        return modulation(embed(stream.noise_level))

    def _denoise(
        self,
        hidden: torch.Tensor,
        *,
        modulation: torch.Tensor,
        rope: RotaryTables | None,
        key_mask: torch.Tensor | None,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the video tower over an already-embedded sequence."""
        for block in self.blocks:
            hidden = block(
                hidden,
                modulation=modulation,
                rope=rope,
                key_mask=key_mask,
                context=context,
                context_mask=context_mask,
            )
        return hidden

    def _video_head(self, hidden: torch.Tensor, stream: TokenStream) -> torch.Tensor:
        """Project the video tower's output back into patch space."""
        predicted = self.final_proj(self.final_norm(hidden))
        # Zero the padding so a downstream reduction that forgets the mask is
        # wrong by nothing rather than by a bias-sized constant per padded token.
        return predicted * stream.mask.unsqueeze(-1).to(predicted.dtype)

    def _null_audio(self, inputs: ModelInput, like: torch.Tensor) -> torch.Tensor:
        """Return correctly shaped zeros for a model that predicts no audio."""
        audio = inputs.audio
        return torch.zeros(
            (audio.batch_size, audio.length, audio.layout.patch_dim),
            dtype=like.dtype,
            device=like.device,
        )

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Predict the per-token field for the video stream.

        Args:
            inputs: Noisy streams plus conditioning. ``inputs.audio`` may be
                zero-length; a plain video model ignores it and returns zeros
                of the matching shape.

        Returns:
            Token-space predictions.
        """
        video = inputs.video
        context, context_mask = self.encode_text(inputs.text)
        hidden = self.patch_embed(video.masked())
        modulation = self._stream_conditioning(
            video, embed=self.time_embed, modulation=self.modulation
        )
        rope = self.rope(video.coords)
        key_mask = key_padding_mask(
            video.mask, assume_dense=self.config.assume_dense_mask
        )
        hidden = self._denoise(
            hidden,
            modulation=modulation,
            rope=rope,
            key_mask=key_mask,
            context=context,
            context_mask=context_mask,
        )
        predicted = self._video_head(hidden, video)
        return ModelOutput(video=predicted, audio=self._null_audio(inputs, predicted))

    def tensor_parallel_plan(
        self,
        *,
        sequence_parallel: bool,
    ) -> tuple[TensorParallelPlan, TensorParallelPlan]:
        """Return the ``(root_plan, block_plan)`` for this architecture.

        Args:
            sequence_parallel: Whether norm and residual regions are sharded
                along the sequence.

        Returns:
            The root and per-block plans.
        """
        root = _root_plan(sequence_parallel=sequence_parallel)
        block = _block_plan(
            sequence_parallel=sequence_parallel,
            cross_attention=self.config.cross_attention,
        )
        return root, block


def _block_plan(
    *, sequence_parallel: bool, cross_attention: bool
) -> TensorParallelPlan:
    """Build the block plan, with the norms emitting local tensors.

    ``SequenceParallel`` defaults to returning a ``DTensor``. Every norm in a
    :class:`~avgen.models.blocks.DiTBlock` is immediately followed by an adaLN
    modulation against a plain, replicated tensor, and mixing a ``DTensor`` with
    a plain tensor in an elementwise op raises. Asking for the local shard
    instead is free — the very next module is a ``PrepareModuleInput`` that
    reconstructs the ``DTensor`` from it.

    Args:
        sequence_parallel: Whether to shard the norm regions.
        cross_attention: Whether blocks carry a text cross-attention branch.

    Returns:
        The block plan.
    """
    # Imported inside the function so that `avgen.models` stays importable
    # without torch.distributed, and because a sharding plan is built once per
    # job rather than on any hot path.
    from torch.distributed.tensor.parallel import SequenceParallel

    from avgen.parallel.tensor import standard_block_plan

    plan = standard_block_plan(
        sequence_parallel=sequence_parallel,
        cross_attention_prefix="cross_attention" if cross_attention else None,
    )
    if sequence_parallel:
        for name in ("attention_norm", "ffn_norm", "cross_norm"):
            if name in plan:
                plan[name] = SequenceParallel(use_local_output=True)
    return plan


def _root_plan(*, sequence_parallel: bool) -> TensorParallelPlan:
    """Build the root plan.

    Args:
        sequence_parallel: Whether the block interior is sequence-sharded.

    Returns:
        The root plan.
    """
    from avgen.parallel.tensor import standard_root_plan

    return standard_root_plan(sequence_parallel=sequence_parallel)


@dataclass(frozen=True, slots=True)
class AVDiTConfig(VideoDiTConfig):
    """Architecture of a joint audio-video diffusion transformer.

    Inherits every video field. Note that ``@dataclass(slots=True)`` rebuilds
    the class object, which breaks zero-argument ``super()``; validation
    therefore calls the base ``__post_init__`` explicitly.

    Args:
        audio_width: Audio-stream residual width. Usually much smaller than the
            video width: an audio latent frame carries far less information than
            a video latent frame, and spending equal width on it is the most
            common way an audio-video model wastes half its parameters.
        audio_num_heads: Audio attention heads.
        audio_in_channels: Audio latent channels.
        audio_out_channels: Predicted audio channels. Defaults to
            ``audio_in_channels``.
        audio_patch_frames: Temporal patch size for audio.
        audio_ffn_hidden: Explicit audio feed-forward inner width.
        audio_rope_time_scale: Multiplier on the audio seconds axis. ``None``
            means share the video model's, which is what puts both streams in
            one rotary phase space and is almost always what you want.
        fusion: One of :data:`FUSION_MODES`.
        fusion_interval: Blocks between fusion points in ``cross_attention``
            mode. Fusing at every block roughly doubles the cross-attention
            cost for a small quality gain; every second block is the usual
            compromise.
        fusion_window_seconds: Half-width of the temporal neighbourhood two
            streams may attend across, in ``cross_attention`` mode. Zero or
            less means unrestricted, which restores the fast attention kernels.

    Raises:
        ValueError: On any invalid or mutually inconsistent field.
    """

    audio_width: int = 512
    audio_num_heads: int = 8
    audio_in_channels: int = 8
    audio_out_channels: int | None = None
    audio_patch_frames: int = 1
    audio_ffn_hidden: int | None = None
    audio_rope_time_scale: float | None = None
    fusion: str = "cross_attention"
    fusion_interval: int = 2
    fusion_window_seconds: float = 0.25

    def __post_init__(self) -> None:
        """Validate the audio stream and the fusion configuration.

        Raises:
            ValueError: Naming the offending field and its value.
        """
        VideoDiTConfig.__post_init__(self)
        for name in (
            "audio_width",
            "audio_num_heads",
            "audio_in_channels",
            "audio_patch_frames",
            "fusion_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.fusion not in FUSION_MODES:
            raise ValueError(
                f"fusion must be one of {FUSION_MODES}; got {self.fusion!r}"
            )
        if self.audio_width % self.audio_num_heads != 0:
            raise ValueError(
                f"audio_width {self.audio_width} must be divisible by "
                f"audio_num_heads {self.audio_num_heads}"
            )
        if self.audio_head_dim % 2 != 0 or self.audio_head_dim < 6:
            raise ValueError(
                f"audio head_dim {self.audio_head_dim} must be even and at least 6"
            )

    @property
    def audio_head_dim(self) -> int:
        """Channels per audio attention head."""
        return self.audio_width // self.audio_num_heads

    @property
    def audio_ffn_width(self) -> int:
        """Audio feed-forward inner width, after rounding."""
        if self.audio_ffn_hidden is not None:
            return self.audio_ffn_hidden
        hidden = int(self.mlp_ratio * self.audio_width)
        multiple = min(self.ffn_multiple_of, max(1, self.audio_width))
        return -(-hidden // multiple) * multiple

    @property
    def audio_patch_dim(self) -> int:
        """Input feature width of one packed audio patch."""
        return self.audio_in_channels * self.audio_patch_frames

    @property
    def audio_out_patch_dim(self) -> int:
        """Predicted feature width of one packed audio patch."""
        channels = (
            self.audio_out_channels
            if self.audio_out_channels is not None
            else self.audio_in_channels
        )
        return channels * self.audio_patch_frames

    @property
    def audio_cond_dim(self) -> int:
        """Width of the audio timestep conditioning vector."""
        return self.width if self.fusion == "joint" else self.audio_width

    @property
    def fusion_points(self) -> tuple[int, ...]:
        """Block indices after which the two towers exchange information."""
        if self.fusion != "cross_attention":
            return ()
        return tuple(
            index
            for index in range(self.depth)
            if (index + 1) % self.fusion_interval == 0
        )


@register_model("av_dit")
class AVDiT(VideoDiT):
    """Joint audio-video diffusion transformer.

    Degenerates exactly to :class:`VideoDiT` when handed a zero-length audio
    stream, which is what lets one checkpoint serve text-to-video,
    video-to-audio, audio-to-video, and joint generation: the task is selected
    by which stream is marked clean in
    :class:`~avgen.core.batch.ConditionMode`, not by which model is loaded.

    Args:
        config: The architecture.
    """

    def __init__(self, config: AVDiTConfig) -> None:
        super().__init__(config)
        self.config: AVDiTConfig = config
        self._audio_patchifier = GridPatchifier(
            patch_frames=config.audio_patch_frames,
            patch_height=1,
            patch_width=1,
            normalize_space=False,
        )
        self.audio_patch_embed = nn.Linear(
            config.audio_patch_dim, config.audio_width, bias=True
        )
        self.audio_time_embed = TimestepEmbedding(
            config.audio_cond_dim, frequency_dim=config.time_frequency_dim
        )
        self.audio_final_norm = rms_norm(config.audio_width, eps=config.norm_eps)
        self.audio_final_proj = nn.Linear(
            config.audio_width, config.audio_out_patch_dim, bias=True
        )

        if config.fusion == "joint":
            # One tower, one sequence. Audio enters and leaves the video width
            # through a pair of projections, which is what keeps asymmetric
            # widths available in a mode that otherwise forces them equal.
            self.audio_in_proj = nn.Linear(config.audio_width, config.width, bias=True)
            self.audio_out_proj = nn.Linear(config.width, config.audio_width, bias=True)
            self.audio_modulation = AdaLNModulation(config.audio_cond_dim, config.width)
            self.audio_blocks = nn.ModuleList()
            self.fusion_blocks = nn.ModuleList()
            self.audio_text_proj: nn.Module = nn.Identity()
            self.audio_rope: nn.Module = nn.Identity()
        else:
            self.audio_in_proj = nn.Identity()  # type: ignore[assignment]
            self.audio_out_proj = nn.Identity()  # type: ignore[assignment]
            self.audio_modulation = AdaLNModulation(
                config.audio_cond_dim, config.audio_width
            )
            self.audio_blocks = nn.ModuleList(
                DiTBlock(
                    config.audio_width,
                    config.audio_num_heads,
                    hidden=config.audio_ffn_width,
                    num_kv_heads=config.num_kv_heads,
                    cross_attention=config.cross_attention,
                    qk_norm=config.qk_norm,
                    eps=config.norm_eps,
                )
                for _ in range(config.depth)
            )
            self.fusion_blocks = nn.ModuleList(
                CrossModalFusion(
                    config.width,
                    config.audio_width,
                    video_heads=config.num_heads,
                    audio_heads=config.audio_num_heads,
                    qk_norm=config.qk_norm,
                    eps=config.norm_eps,
                )
                for _ in config.fusion_points
            )
            self.audio_text_proj = nn.Linear(
                config.text_width, config.audio_width, bias=True
            )
            self.audio_rope = RotaryEmbedding(
                config.audio_head_dim,
                theta=config.rope_theta,
                time_scale=(
                    config.audio_rope_time_scale
                    if config.audio_rope_time_scale is not None
                    else config.rope_time_scale
                ),
                scaling=config.rope_scaling,
            )

    @property
    def audio_patchifier(self) -> Patchifier:
        """The patchifier the audio stream must be built with."""
        return self._audio_patchifier

    def init_weights(self) -> None:
        """Initialise the video tower, the audio tower, and the fusion."""
        super().init_weights()
        config = self.config
        init_linear(self.audio_patch_embed, std=config.init_std)
        self.audio_time_embed.init_weights()
        self.audio_modulation.init_weights()
        init_norm(self.audio_final_norm)
        init_linear(self.audio_final_proj, zero=True)
        for block in self.audio_blocks:
            block.init_weights(depth=config.depth, std=config.init_std)
        for fusion in self.fusion_blocks:
            fusion.init_weights(std=config.init_std)
        if isinstance(self.audio_in_proj, nn.Linear):
            init_linear(self.audio_in_proj, std=config.init_std)
        if isinstance(self.audio_out_proj, nn.Linear):
            init_linear(self.audio_out_proj, std=config.init_std)
        if isinstance(self.audio_text_proj, nn.Linear):
            init_linear(self.audio_text_proj, std=config.init_std)

    def model_shape(
        self,
        *,
        sequence_length: int,
        micro_batch_size: int = 1,
    ) -> ModelShape:
        """Describe the model to the simulators.

        ``sequence_length`` is the **combined** token count, because that is
        what both fusion modes actually attend over and therefore what drives
        activation memory.

        Args:
            sequence_length: Video plus audio tokens per sample.
            micro_batch_size: Samples per rank per microbatch.

        Returns:
            The shape the simulator prices.
        """
        return super().model_shape(
            sequence_length=sequence_length, micro_batch_size=micro_batch_size
        )

    def _audio_context(
        self, text: TextContext
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Project the text context into the audio tower's width."""
        if text.is_empty or not self.config.cross_attention:
            return None, None
        if not isinstance(self.audio_text_proj, nn.Linear):
            return None, None
        mask = key_padding_mask(text.mask, assume_dense=self.config.assume_dense_mask)
        return self.audio_text_proj(text.features), mask

    def _forward_joint(self, inputs: ModelInput) -> ModelOutput:
        """Run both streams through one self-attention tower."""
        video, audio = inputs.video, inputs.audio
        context, context_mask = self.encode_text(inputs.text)

        hidden = torch.cat(
            (
                self.patch_embed(video.masked()),
                self.audio_in_proj(self.audio_patch_embed(audio.masked())),
            ),
            dim=1,
        )
        # The two streams carry different noise levels, so the modulation has to
        # vary along the sequence even when each stream's own level is a scalar.
        # This is the joint mode's real cost: (batch, length, 6, width) rather
        # than (batch, 6, width), live for the whole forward.
        video_mod = self.modulation(self.time_embed(video.noise_level))
        audio_mod = self.audio_modulation(self.audio_time_embed(audio.noise_level))
        modulation = torch.cat(
            (
                _expand_modulation(video_mod, video.length),
                _expand_modulation(audio_mod, audio.length),
            ),
            dim=1,
        )
        # One rotary phase space over both streams. Audio coordinates carry row
        # and column zero, so the spatial rotations are the identity there and
        # only the shared seconds axis links the modalities — which is exactly
        # the invariant audio-video synchronisation needs.
        rope = self.rope(torch.cat((video.coords, audio.coords), dim=1))
        key_mask = key_padding_mask(
            torch.cat((video.mask, audio.mask), dim=1),
            assume_dense=self.config.assume_dense_mask,
        )
        hidden = self._denoise(
            hidden,
            modulation=modulation,
            rope=rope,
            key_mask=key_mask,
            context=context,
            context_mask=context_mask,
        )
        video_hidden, audio_hidden = hidden.split((video.length, audio.length), dim=1)
        predicted_audio = self.audio_final_proj(
            self.audio_final_norm(self.audio_out_proj(audio_hidden))
        )
        predicted_audio = predicted_audio * audio.mask.unsqueeze(-1).to(
            predicted_audio.dtype
        )
        return ModelOutput(
            video=self._video_head(video_hidden, video), audio=predicted_audio
        )

    def _forward_two_tower(self, inputs: ModelInput) -> ModelOutput:
        """Run two towers with periodic temporally-local cross-attention."""
        config = self.config
        video, audio = inputs.video, inputs.audio
        context, context_mask = self.encode_text(inputs.text)
        audio_context, audio_context_mask = self._audio_context(inputs.text)

        video_hidden = self.patch_embed(video.masked())
        audio_hidden = self.audio_patch_embed(audio.masked())
        video_mod = self.modulation(self.time_embed(video.noise_level))
        audio_mod = self.audio_modulation(self.audio_time_embed(audio.noise_level))
        video_rope = self.rope(video.coords)
        audio_rope = self.audio_rope(audio.coords)
        video_key_mask = key_padding_mask(
            video.mask, assume_dense=config.assume_dense_mask
        )
        audio_key_mask = key_padding_mask(
            audio.mask, assume_dense=config.assume_dense_mask
        )

        # Built once and reused at every fusion point: the coordinates do not
        # change between blocks, and rebuilding a dense (queries, keys) mask per
        # fusion point is how this architecture runs out of memory.
        video_to_audio = temporal_neighbour_mask(
            video.coords[..., 0],
            audio.coords[..., 0],
            window_seconds=config.fusion_window_seconds,
            key_mask=audio.mask,
        )
        audio_to_video = temporal_neighbour_mask(
            audio.coords[..., 0],
            video.coords[..., 0],
            window_seconds=config.fusion_window_seconds,
            key_mask=video.mask,
        )

        fusion_at = {point: index for index, point in enumerate(config.fusion_points)}
        for index in range(config.depth):
            video_hidden = self.blocks[index](
                video_hidden,
                modulation=video_mod,
                rope=video_rope,
                key_mask=video_key_mask,
                context=context,
                context_mask=context_mask,
            )
            audio_hidden = self.audio_blocks[index](
                audio_hidden,
                modulation=audio_mod,
                rope=audio_rope,
                key_mask=audio_key_mask,
                context=audio_context,
                context_mask=audio_context_mask,
            )
            fusion_index = fusion_at.get(index)
            if fusion_index is not None:
                video_hidden, audio_hidden = self.fusion_blocks[fusion_index](
                    video_hidden,
                    audio_hidden,
                    video_to_audio=video_to_audio,
                    audio_to_video=audio_to_video,
                )

        predicted_audio = self.audio_final_proj(self.audio_final_norm(audio_hidden))
        predicted_audio = predicted_audio * audio.mask.unsqueeze(-1).to(
            predicted_audio.dtype
        )
        return ModelOutput(
            video=self._video_head(video_hidden, video), audio=predicted_audio
        )

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Predict the per-token field for both streams.

        Args:
            inputs: Noisy streams plus conditioning.

        Returns:
            Token-space predictions for video and audio.
        """
        if not inputs.has_audio:
            # No audio tokens: the audio tower has nothing to consume and every
            # fusion point would be a no-op, so this is exactly the video model.
            return super().forward(inputs)
        if self.config.fusion == "joint":
            return self._forward_joint(inputs)
        return self._forward_two_tower(inputs)

    def tensor_parallel_plan(
        self,
        *,
        sequence_parallel: bool,
    ) -> tuple[TensorParallelPlan, TensorParallelPlan]:
        """Return the ``(root_plan, block_plan)`` for this architecture.

        The audio tower is not an ``nn.ModuleList`` the framework knows about,
        so its blocks are addressed by fully-qualified name from the root plan.

        Args:
            sequence_parallel: Whether norm and residual regions are sharded
                along the sequence.

        Returns:
            The root and per-block plans.

        Raises:
            ValueError: If sequence-parallel tensor parallelism is requested for
                ``cross_attention`` fusion. Cross-modal attention needs the
                whole of the *other* stream as keys, and a sequence-sharded
                stream would silently give it a fraction of one — a wrong
                answer, not a slow one. Shard the sequence with context
                parallelism instead, which shards the token streams themselves.
        """
        if sequence_parallel and self.config.fusion == "cross_attention":
            raise ValueError(
                "AVDiT(fusion='cross_attention') does not support sequence-parallel "
                "tensor parallelism: the fusion blocks attend across the whole of "
                "the other stream, which a sequence shard cannot provide. Use "
                "ParallelConfig(sequence_parallel=False), or shard the sequence "
                "with context parallelism (ParallelDims(context=...))."
            )
        root, block = super().tensor_parallel_plan(sequence_parallel=sequence_parallel)
        if self.config.fusion == "joint":
            return root, block

        audio_plan = _block_plan(
            sequence_parallel=False, cross_attention=self.config.cross_attention
        )
        for index in range(self.config.depth):
            for name, style in audio_plan.items():
                root[f"audio_blocks.{index}.{name}"] = style
        return root, block


def _expand_modulation(modulation: torch.Tensor, length: int) -> torch.Tensor:
    """Broadcast a per-sample modulation to per-token, for a joint sequence.

    Args:
        modulation: ``(batch, 6, width)`` or ``(batch, length, 6, width)``.
        length: Target sequence length.

    Returns:
        ``(batch, length, 6, width)``.

    Raises:
        ValueError: If a per-token modulation does not already have ``length``
            entries.
    """
    if modulation.ndim == 4:
        if modulation.shape[1] != length:
            raise ValueError(
                f"per-token modulation covers {modulation.shape[1]} tokens but the "
                f"stream holds {length}"
            )
        return modulation
    return modulation[:, None].expand(-1, length, -1, -1)


_PRESETS: dict[str, VideoDiTConfig] = {
    # CPU-sized. Every dimension is the smallest legal value, so a full forward
    # runs in milliseconds and a unit test can assert on real numbers.
    "tiny": VideoDiTConfig(
        width=64,
        depth=2,
        num_heads=4,
        ffn_hidden=128,
        in_channels=4,
        text_width=32,
        text_context_length=8,
        text_refiner_depth=1,
        cond_width=64,
        time_frequency_dim=32,
    ),
    "tiny_av": AVDiTConfig(
        width=64,
        depth=2,
        num_heads=4,
        ffn_hidden=128,
        in_channels=4,
        text_width=32,
        text_context_length=8,
        text_refiner_depth=1,
        cond_width=64,
        time_frequency_dim=32,
        audio_width=32,
        audio_num_heads=2,
        audio_in_channels=4,
        audio_ffn_hidden=64,
        fusion_interval=1,
    ),
    # Single-node scale: fits on one 80 GB device with room for a real batch.
    "dit_300m": VideoDiTConfig(
        width=768, depth=24, num_heads=12, text_context_length=256
    ),
    # The default production size. 2048/16 gives a 128-wide head, which is the
    # dimension every fused attention kernel is tuned for.
    "dit_2b": VideoDiTConfig(
        width=2048, depth=24, num_heads=16, text_context_length=512
    ),
    "dit_5b": VideoDiTConfig(
        width=3072, depth=28, num_heads=24, text_context_length=512
    ),
    # Wide before deep: at fixed parameter count width parallelises better
    # (tensor parallelism shards it) while depth only adds pipeline stages.
    "dit_14b": VideoDiTConfig(
        width=4096, depth=42, num_heads=32, text_context_length=512
    ),
}


def preset(name: str) -> VideoDiTConfig:
    """Return a named architecture preset.

    Args:
        name: Preset key.

    Returns:
        The configuration. Presets are frozen dataclasses, so the returned
        object is safe to share; use :func:`dataclasses.replace` to vary one
        field.

    Raises:
        KeyError: If the preset does not exist.
    """
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(sorted(_PRESETS))}"
        ) from None


def list_presets() -> tuple[str, ...]:
    """List every named architecture preset.

    Returns:
        Sorted preset keys.
    """
    return tuple(sorted(_PRESETS))
