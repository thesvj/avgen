"""Transformer blocks: the repeated unit the parallelism layer addresses.

:class:`DiTBlock` is a pre-norm block with adaLN-single modulation and optional
text cross-attention. Its submodule names are **not** free: ``attention_norm``,
``attention``, ``cross_norm``, ``cross_attention``, ``ffn_norm``, and
``feed_forward`` are the exact names that
:func:`~avgen.parallel.tensor.standard_block_plan` addresses, and the positional
arity of each submodule's ``forward`` is what its ``PrepareModuleInput`` entries
describe. Renaming a submodule or moving an argument from positional to keyword
does not fail loudly — it produces a model that shards into a plausible-looking
but numerically wrong result. Treat the names and the signatures as the API they
are.

**Pre-norm, not post-norm.** Post-norm needs learning-rate warmup and careful
initialisation to stay stable past about twenty layers; pre-norm keeps a clean
residual path from input to output and trains at depth 40 without ceremony. The
price is that the residual stream's variance grows with depth, which is why the
output projections are initialised with a depth-scaled standard deviation.

**Cross-attention is ungated.** adaLN-single emits six vectors — shift, scale,
gate for attention and the same for the feed-forward — and cross-attention gets
none of them. Text conditioning should not be modulated by the noise level; a
gate that can close would let the model learn to ignore the prompt at high noise,
which is precisely where the prompt matters most. The branch still starts as the
identity, via a zero-initialised output projection.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from avgen.models.layers import (
    MODULATION_CHUNKS,
    CrossAttention,
    SelfAttention,
    SwiGLU,
    init_norm,
    modulate,
    rms_norm,
)
from avgen.models.rope import RotaryTables

__all__ = [
    "CrossModalFusion",
    "DiTBlock",
    "TextRefinerBlock",
    "temporal_neighbour_mask",
]


def temporal_neighbour_mask(
    query_seconds: torch.Tensor,
    key_seconds: torch.Tensor,
    *,
    window_seconds: float,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build a mask restricting attention to temporally nearby tokens.

    Audio-video correspondence is local in time and only in time: a footstep
    matches the frame it lands on, not the frame ten seconds later. Letting every
    audio token attend to every video token spends quadratic compute learning to
    ignore almost all of it, and in practice yields a model whose audio drifts
    out of sync because nothing in the architecture says it should not.

    The window is expressed in **seconds** and compared against the physical
    time coordinates the token streams carry, so it means the same thing at any
    frame rate and for streams whose rates differ by an order of magnitude.

    **Cost warning.** The result is a dense ``(batch, 1, queries, keys)`` boolean
    tensor. For 100k video tokens against 2k audio tokens that is 200M elements
    per sample. Fusing at every block instead of every few blocks, or fusing full
    resolution video against full resolution audio, is how this becomes the
    dominant memory term. Pass ``window_seconds <= 0`` to disable masking
    entirely and get the dense fast path back.

    Args:
        query_seconds: ``(batch, queries)`` physical time of each query token.
        key_seconds: ``(batch, keys)`` physical time of each key token.
        window_seconds: Half-width of the admissible time difference.
        key_mask: Optional ``(batch, keys)`` key validity.

    Returns:
        A ``(batch, 1, queries, keys)`` boolean mask, or ``None`` when no
        restriction applies and the fast kernels should be left alone.
    """
    if window_seconds <= 0.0:
        if key_mask is None or bool(key_mask.all()):
            return None
        return key_mask[:, None, None, :]

    delta = (query_seconds[:, :, None] - key_seconds[:, None, :]).abs()
    mask = delta <= window_seconds
    if key_mask is not None:
        mask = mask & key_mask[:, None, :]
    # A query with no admissible key would make softmax divide by zero and emit
    # NaN across the whole row. Falling back to unrestricted attention for those
    # queries is both finite and the sensible behaviour: a token with no local
    # partner has nothing local to look at.
    mask = mask | ~mask.any(dim=-1, keepdim=True)
    return mask[:, None]


class DiTBlock(nn.Module):
    """One pre-norm diffusion transformer block over a token sequence.

    Operates purely on ``(batch, length, width)``. It knows nothing about
    frames, resolution, or modality: all of that reaches it through the rotary
    tables and the mask, which is what lets the same block serve video, audio,
    and a concatenated joint sequence without a variant per case.

    Args:
        width: Model width.
        num_heads: Query heads.
        hidden: Feed-forward inner width.
        head_dim: Channels per head. Defaults to ``width // num_heads``.
        num_kv_heads: Key/value heads for grouped-query attention.
        context_width: Width of the cross-attention context. Defaults to
            ``width``.
        cross_attention: Whether to include the text cross-attention branch.
        qk_norm: Whether to RMS-normalise queries and keys.
        eps: Normalisation epsilon.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        *,
        hidden: int,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        context_width: int | None = None,
        cross_attention: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = width
        self.attention_norm = rms_norm(width, eps=eps)
        self.attention = SelfAttention(
            width,
            num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.cross_norm: nn.Module | None = None
        self.cross_attention: CrossAttention | None = None
        if cross_attention:
            self.cross_norm = rms_norm(width, eps=eps)
            self.cross_attention = CrossAttention(
                width,
                num_heads,
                context_width=context_width,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                qk_norm=qk_norm,
                eps=eps,
            )
        self.ffn_norm = rms_norm(width, eps=eps)
        self.feed_forward = SwiGLU(width, hidden)
        # adaLN-single's per-block correction to the shared modulation. Zero, so
        # that a fresh block is exactly the identity at any depth; PixArt
        # initialises it randomly, which trades identity-at-init for a slightly
        # faster first few hundred steps and is not worth the warmup fragility.
        self.scale_shift_table = nn.Parameter(torch.zeros(MODULATION_CHUNKS, width))

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        modulation: torch.Tensor,
        rope: RotaryTables | None = None,
        key_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one block.

        Args:
            hidden: ``(batch, length, width)`` activations.
            modulation: ``(batch, 6, width)`` per-sample or
                ``(batch, length, 6, width)`` per-token modulation from
                :class:`~avgen.models.layers.AdaLNModulation`.
            rope: Rotary tables for this rank's coordinate shard.
            key_mask: SDPA mask for self-attention, or ``None``.
            context: ``(batch, context_length, context_width)`` cross-attention
                context, or ``None`` to skip the branch.
            context_mask: Mask for the cross-attention context.

        Returns:
            ``(batch, length, width)`` activations.

        Raises:
            ValueError: If ``modulation`` has an unexpected rank.
        """
        if modulation.ndim not in (3, 4):
            raise ValueError(
                f"modulation must be (batch, 6, width) or (batch, length, 6, width); "
                f"got {tuple(modulation.shape)}"
            )
        combined = modulation + self.scale_shift_table
        parts = combined.unbind(dim=-2)
        if combined.ndim == 3:
            # Per-sample modulation needs a length axis to broadcast over.
            parts = tuple(part[:, None] for part in parts)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = parts

        normed = modulate(self.attention_norm(hidden), shift_attn, scale_attn)
        hidden = hidden + gate_attn * self.attention(normed, rope, key_mask=key_mask)

        cross_norm = self.cross_norm
        cross_attention = self.cross_attention
        if (
            context is not None
            and cross_norm is not None
            and cross_attention is not None
        ):
            hidden = hidden + cross_attention(cross_norm(hidden), context, context_mask)

        normed = modulate(self.ffn_norm(hidden), shift_ffn, scale_ffn)
        return hidden + gate_ffn * self.feed_forward(normed)

    def init_weights(self, *, depth: int = 1, std: float = 0.02) -> None:
        """Initialise every leaf in this block in place.

        Args:
            depth: Total number of blocks in the model. Output projections are
                scaled by ``1 / sqrt(2 * depth)`` so that the residual stream's
                variance stays bounded as depth grows; without it a 40-block
                model's activations at the final norm are an order of magnitude
                larger than a 4-block model's, and the same learning rate cannot
                serve both.
            std: Base standard deviation.
        """
        out_std = std / math.sqrt(2.0 * max(1, depth))
        init_norm(self.attention_norm)
        init_norm(self.ffn_norm)
        self.attention.init_weights(std=std, out_std=out_std)
        self.feed_forward.init_weights(std=std, out_std=out_std)
        if self.cross_attention is not None and self.cross_norm is not None:
            init_norm(self.cross_norm)
            self.cross_attention.init_weights(std=std)
        nn.init.zeros_(self.scale_shift_table)


class TextRefinerBlock(nn.Module):
    """A bidirectional block that reworks frozen text features before use.

    Text towers are trained for a language objective — causal prediction, span
    corruption — and their hidden states are organised for that, not for telling
    a video model what to draw. A handful of trainable bidirectional blocks over
    the prompt (LTX-Video's connector, HunyuanVideo's token refiner) lets the
    model reorganise the prompt into something cross-attention can index
    cheaply, and it costs almost nothing: the prompt is a few hundred tokens
    against a hundred thousand video tokens.

    Deliberately unmodulated. The refined prompt is the same at every noise
    level, so it can be computed once per sample and reused across every
    sampling step; making it depend on the timestep would multiply inference
    text cost by the step count for no observed gain.

    Args:
        width: Text feature width after projection into the model.
        num_heads: Attention heads.
        hidden: Feed-forward inner width.
        qk_norm: Whether to RMS-normalise queries and keys.
        eps: Normalisation epsilon.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        *,
        hidden: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attention_norm = rms_norm(width, eps=eps)
        self.attention = SelfAttention(width, num_heads, qk_norm=qk_norm, eps=eps)
        self.ffn_norm = rms_norm(width, eps=eps)
        self.feed_forward = SwiGLU(width, hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine the text features.

        Args:
            hidden: ``(batch, tokens, width)`` projected text features.
            key_mask: SDPA mask over the prompt tokens, or ``None``.

        Returns:
            ``(batch, tokens, width)`` refined features.
        """
        hidden = hidden + self.attention(
            self.attention_norm(hidden), None, key_mask=key_mask
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))

    def init_weights(self, *, depth: int = 1, std: float = 0.02) -> None:
        """Initialise every leaf in this block in place.

        Args:
            depth: Number of refiner blocks, for residual scaling.
            std: Base standard deviation.
        """
        out_std = std / math.sqrt(2.0 * max(1, depth))
        init_norm(self.attention_norm)
        init_norm(self.ffn_norm)
        self.attention.init_weights(std=std, out_std=out_std)
        self.feed_forward.init_weights(std=std, out_std=out_std)


class CrossModalFusion(nn.Module):
    """Bidirectional cross-attention between the video and audio streams.

    Each stream keeps its own tower, its own width, and its own tokens; fusion
    happens periodically through a pair of cross-attention branches, one in each
    direction. Both branches are ungated with zero-initialised output
    projections, so a fresh model is two independent unimodal towers that learn
    to talk to each other rather than starting from an entangled state.

    The alternative — concatenating the streams into one self-attention — is
    also shipped, as :class:`~avgen.models.dit.AVDiT`'s ``joint`` mode. This one
    wins when the streams want different widths and different depths of
    processing; the joint one wins when they should share parameters.

    Args:
        video_width: Video-stream width.
        audio_width: Audio-stream width.
        video_heads: Heads for the video-side query.
        audio_heads: Heads for the audio-side query.
        qk_norm: Whether to RMS-normalise queries and keys.
        eps: Normalisation epsilon.
    """

    def __init__(
        self,
        video_width: int,
        audio_width: int,
        *,
        video_heads: int,
        audio_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.video_norm = rms_norm(video_width, eps=eps)
        self.video_from_audio = CrossAttention(
            video_width,
            video_heads,
            context_width=audio_width,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.audio_norm = rms_norm(audio_width, eps=eps)
        self.audio_from_video = CrossAttention(
            audio_width,
            audio_heads,
            context_width=video_width,
            qk_norm=qk_norm,
            eps=eps,
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        *,
        video_to_audio: torch.Tensor | None = None,
        audio_to_video: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange information between the two streams.

        Both directions read the *pre-fusion* activations, so the update is
        symmetric. Updating video first and letting audio read the updated video
        would make the result depend on an arbitrary ordering, and would break
        the equivalence between the video-to-audio and audio-to-video tasks that
        a single joint model is supposed to preserve.

        Args:
            video: ``(batch, video_length, video_width)`` activations.
            audio: ``(batch, audio_length, audio_width)`` activations.
            video_to_audio: Mask for video queries attending to audio keys.
            audio_to_video: Mask for audio queries attending to video keys.

        Returns:
            The updated video and audio activations.
        """
        video_update = self.video_from_audio(
            self.video_norm(video), audio, video_to_audio
        )
        audio_update = self.audio_from_video(
            self.audio_norm(audio), video, audio_to_video
        )
        return video + video_update, audio + audio_update

    def init_weights(self, *, std: float = 0.02) -> None:
        """Initialise both directions in place.

        Args:
            std: Base standard deviation.
        """
        init_norm(self.video_norm)
        init_norm(self.audio_norm)
        self.video_from_audio.init_weights(std=std)
        self.audio_from_video.init_weights(std=std)
