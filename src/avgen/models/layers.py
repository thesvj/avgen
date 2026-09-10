"""The primitive layers a video diffusion transformer is assembled from.

Four decisions in this file are load-bearing and worth reading before changing
anything.

**Separate q/k/v projections, not a fused one.** A fused ``qkv`` matmul is
measurably faster on a single GPU. It is also unshardable by the standard
tensor-parallel plan, which addresses ``q_proj``, ``k_proj``, and ``v_proj`` by
name and column-shards each one independently — a fused projection would have to
be split at head boundaries inside one weight, and grouped-query attention makes
those boundaries uneven. Three matmuls cost a few percent; a wrong shard costs a
training run.

**Head counts are read from the runtime tensor, never from configuration.**
Column-wise tensor parallelism gives each rank a slice of the heads, so a module
that reshapes using its configured head count produces garbage the moment
``tp > 1``. Dividing the observed projection width by the (shard-invariant) head
dimension is correct at any degree, with or without parallelism.

**QK-RMSNorm.** Attention logits grow with width and depth, and in bf16 a large
logit saturates the softmax into a one-hot that produces no gradient. The
failure looks like a loss that plateaus rather than diverges, which is why it
goes undiagnosed for days. Normalising queries and keys per head before the dot
product bounds the logit scale and costs two cheap kernels.

**Zero-initialised gates (DiT-zero).** Every residual branch is multiplied by a
gate that starts at exactly zero, so a freshly built model of any depth is the
identity function and its output is exactly zero. The alternative — small random
init — makes deep diffusion transformers need learning-rate warmup tricks to
avoid an early loss spike. Note that the gate and the branch's output projection
must not *both* be zero: the gradient of the branch weights is proportional to
the gate, so a doubly-zeroed branch is dead forever. Gated branches zero the
gate; ungated branches (cross-attention, the final head) zero the projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from avgen.models.attention import attention
from avgen.models.rope import RotaryTables, apply_rotary

__all__ = [
    "MODULATION_CHUNKS",
    "AdaLNModulation",
    "CrossAttention",
    "FusedRMSNorm",
    "SelfAttention",
    "SwiGLU",
    "TimestepEmbedding",
    "init_linear",
    "init_norm",
    "match_dtype",
    "modulate",
    "rms_norm",
    "sinusoidal_embedding",
]

#: adaLN-single emits shift, scale, and gate for the attention branch and the
#: same three for the feed-forward branch. Cross-attention is deliberately
#: ungated (see :class:`~avgen.models.blocks.DiTBlock`), so the count is six and
#: not nine.
MODULATION_CHUNKS = 6


class FusedRMSNorm(nn.RMSNorm):
    """RMSNorm that keeps its fused kernel under autocast.

    ``torch.rms_norm`` only dispatches to its fused implementation when the
    input and the weight share a dtype. Under ``torch.autocast`` the activation
    arrives as bfloat16 while the parameter stays float32, the dispatch falls
    back to a composite of elementwise ops, and the layer becomes roughly ten
    times slower — measured at 23.1 ms against 2.4 ms for a
    ``(4, 16384, 2048)`` activation on an Ada card. At three norms per block
    that is seconds per forward pass on a deep model.

    FSDP2 hides this, because its mixed-precision policy casts parameters to the
    compute dtype before the forward runs. Single-device training, gradient-
    accumulation microbatches on one GPU, and evaluation all use plain autocast,
    where nothing casts the weight — so the slow path is exactly the one people
    hit while developing, and the fast path is the one they benchmark.

    Casting the weight to the activation dtype restores the fused kernel. The
    cast is a no-op when the dtypes already agree, so the FSDP path is untouched.

    **This is not numerically free, and the trade is deliberate.** Rounding the
    gain to bfloat16 roughly doubles the error against an fp32 reference — from
    about four bfloat16 units in the last place to about eight, measured at
    0.0155 against 0.0300 on a unit-scale activation. That is the correct trade
    for two reasons. First, it is what already happens in any distributed run:
    FSDP2's mixed-precision policy stores the parameter in bfloat16, so the
    fused path is what production numerics actually are, and the fp32-weight
    path was the anomaly. Second, a norm gain is a learned scale near one, where
    a 0.4% relative perturbation is far inside what the optimizer corrects
    within a few steps.

    If you need the fp32-weight numerics — a bit-exact comparison against a
    reference implementation, say — run the model outside autocast, where the
    activation is fp32 and the cast does nothing.
    """

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalise ``hidden``, matching the weight dtype to the input.

        Args:
            hidden: Activation to normalise.

        Returns:
            The normalised activation.
        """
        weight = self.weight
        if weight is not None and weight.dtype != hidden.dtype:
            weight = weight.to(hidden.dtype)
        return F.rms_norm(hidden, self.normalized_shape, weight, self.eps)


def match_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast a tensor to the dtype of a module's weights.

    Under FSDP2 the parameters sit in the compute dtype while the batch arrives
    in float32, and avgen deliberately does not let FSDP cast forward inputs
    wholesale — coordinates and noise levels are physical quantities that lose
    meaning in bfloat16 (see
    :meth:`avgen.parallel.precision.PrecisionConfig.fsdp_policy`). So the cast
    happens here instead: once, at each point where an externally-supplied
    tensor meets a projection, and never on the metadata.

    The cast is a no-op when the dtypes already agree, so the single-device and
    autocast paths are unaffected.

    Args:
        tensor: The externally-supplied tensor.
        module: The module whose weight dtype to match.

    Returns:
        The tensor in the module's weight dtype.

    Raises:
        AttributeError: If the module exposes no ``weight``.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        raise AttributeError(
            f"{type(module).__name__} has no weight to match a dtype against"
        )
    return tensor if tensor.dtype == weight.dtype else tensor.to(weight.dtype)


def rms_norm(width: int, *, eps: float = 1e-6) -> nn.RMSNorm:
    """Build the normalisation layer used everywhere in the model.

    RMSNorm rather than LayerNorm: it drops the mean subtraction, which is one
    fewer reduction over a ``(batch, 100k, width)`` activation, and matches the
    normalisation every recent video and language tower uses.

    :class:`FusedRMSNorm` rather than ``nn.RMSNorm`` directly: it subclasses it,
    so ``SequenceParallel`` still knows how to shard it and the state-dict keys
    are unchanged, but it keeps the fused kernel under autocast. See that class
    for the measurement.

    Args:
        width: Normalised feature width.
        eps: Numerical floor. Deliberately larger than the LayerNorm default,
            because bf16 activations can produce a mean square small enough that
            ``1e-12`` underflows the reciprocal square root.

    Returns:
        The normalisation module.
    """
    return FusedRMSNorm(width, eps=eps)


def init_linear(layer: nn.Linear, *, std: float = 0.02, zero: bool = False) -> None:
    """Initialise a linear layer in place, meta-device safe.

    Args:
        layer: The layer to initialise. Must already hold real storage; on a
            meta-built model call this only after ``to_empty()``.
        std: Standard deviation of the truncated normal used for the weight.
        zero: Whether to zero the weight instead, for a DiT-zero branch.
    """
    if zero:
        nn.init.zeros_(layer.weight)
    else:
        # Truncated at two sigma: an untruncated normal occasionally draws a
        # weight large enough to saturate a bf16 activation on step zero.
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_norm(norm: nn.Module) -> None:
    """Initialise a normalisation layer's affine weight to one, in place.

    Args:
        norm: A module exposing an optional ``weight`` parameter.
    """
    weight = getattr(norm, "weight", None)
    if isinstance(weight, torch.Tensor):
        nn.init.ones_(weight)
    bias = getattr(norm, "bias", None)
    if isinstance(bias, torch.Tensor):
        nn.init.zeros_(bias)


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply an adaLN shift and scale.

    The scale is applied as ``1 + scale`` rather than ``scale`` so that a zero
    modulation is the identity, which is what lets the modulation projection be
    zero-initialised without disabling the normalised activation entirely.

    Args:
        x: ``(batch, length, width)`` normalised activations.
        shift: ``(batch, 1, width)`` or ``(batch, length, width)`` shift.
        scale: Same shape as ``shift``.

    Returns:
        The modulated activations.
    """
    return x * (1.0 + scale) + shift


def sinusoidal_embedding(
    values: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10000.0,
    scale: float = 1000.0,
) -> torch.Tensor:
    """Embed continuous noise levels as a bank of sinusoids.

    Args:
        values: Noise levels in ``[0, 1]``, of any shape.
        dim: Embedding width. Must be even.
        max_period: Longest sinusoid period.
        scale: Multiplier applied to the noise level first. Diffusion models
            conventionally embed a timestep in ``[0, 1000]``; flow matching
            works in ``[0, 1]``, and embedding those raw would confine every
            sinusoid to a fraction of one period, leaving nearby noise levels
            nearly indistinguishable. Rescaling recovers the resolution and
            keeps checkpoints comparable with the diffusion literature.

    Returns:
        ``values.shape + (dim,)`` float32 embeddings.

    Raises:
        ValueError: If ``dim`` is not a positive even integer.
    """
    if dim < 2 or dim % 2 != 0:
        raise ValueError(f"embedding dim must be a positive even integer; got {dim!r}")
    half = dim // 2
    exponent = torch.arange(half, device=values.device, dtype=torch.float32) / half
    # Computed in float32 always: the whole point of the embedding is to
    # separate neighbouring noise levels, and bf16 cannot resolve 0.501 from
    # 0.502 once the values are scaled up.
    frequencies = torch.exp(-math.log(max_period) * exponent)
    angles = values.float()[..., None] * scale * frequencies
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class TimestepEmbedding(nn.Module):
    """Maps a noise level to the conditioning vector adaLN consumes.

    Accepts both shapes that :class:`~avgen.core.tokens.TokenStream` allows:

    * ``(batch,)`` — one noise level per sample. This is the fast path and the
      one to use for ordinary generation. The resulting conditioning is
      ``(batch, width)``, a few kilobytes, broadcast across the whole sequence.
    * ``(batch, length)`` — one noise level per token. This is what makes
      inpainting and continuation *exact*: a clean anchor token is genuinely
      labelled ``t = 0`` instead of being handed to the model at the sequence's
      average noise level and marked with a mask that the model has to learn to
      trust. The cost is real and should be budgeted: conditioning becomes
      ``(batch, length, width)`` and the modulation derived from it becomes
      ``(batch, length, 6 * width)``, which at 100k tokens and width 2048 is
      about 2.5 GB in bf16 per microbatch, live for the whole forward.

    Args:
        width: Output conditioning width.
        frequency_dim: Width of the sinusoidal bank feeding the MLP.
        max_period: Longest sinusoid period.
        scale: Noise-level rescaling before the sinusoids.
    """

    def __init__(
        self,
        width: int,
        *,
        frequency_dim: int = 256,
        max_period: float = 10000.0,
        scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.scale = scale
        self.in_proj = nn.Linear(frequency_dim, width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Embed a per-sample or per-token noise level.

        Args:
            noise_level: ``(batch,)`` or ``(batch, length)`` float32 in
                ``[0, 1]``.

        Returns:
            ``(batch, width)`` or ``(batch, length, width)`` conditioning.
        """
        features = sinusoidal_embedding(
            noise_level,
            self.frequency_dim,
            max_period=self.max_period,
            scale=self.scale,
        ).to(self.in_proj.weight.dtype)
        return self.out_proj(F.silu(self.in_proj(features)))

    def init_weights(self) -> None:
        """Initialise the projections in place."""
        init_linear(self.in_proj)
        init_linear(self.out_proj)


class AdaLNModulation(nn.Module):
    """The shared adaLN-single MLP producing six modulation vectors.

    This is PixArt's adaLN-single rather than DiT's per-block modulation. DiT
    gives every block its own ``cond -> 6 * width`` projection, which at depth
    28 and width 2048 is roughly 700M parameters spent entirely on conditioning
    — more than a quarter of a 2B model. adaLN-single keeps **one** projection
    at the root and gives each block a small learned table that is added to it,
    which reproduces the quality at a fraction of the parameter cost and, more
    usefully here, means the expensive ``(batch, length, 6, width)`` tensor is
    computed once per forward instead of once per block.

    Zero-initialised, so a fresh model produces zero shift, zero scale (hence
    unit gain through :func:`modulate`), and a zero gate.

    Args:
        cond_width: Width of the conditioning vector.
        width: Model width being modulated.
        chunks: Number of modulation vectors. Six unless a block shape changes.
    """

    def __init__(
        self,
        cond_width: int,
        width: int,
        *,
        chunks: int = MODULATION_CHUNKS,
    ) -> None:
        super().__init__()
        self.width = width
        self.chunks = chunks
        self.proj = nn.Linear(cond_width, chunks * width, bias=True)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Project conditioning into stacked modulation vectors.

        Args:
            cond: ``(batch, cond_width)`` or ``(batch, length, cond_width)``.

        Returns:
            ``(batch, chunks, width)`` or ``(batch, length, chunks, width)``.
        """
        return self.proj(F.silu(cond)).unflatten(-1, (self.chunks, self.width))

    def init_weights(self) -> None:
        """Zero the projection so a fresh model is the identity."""
        init_linear(self.proj, zero=True)


class SwiGLU(nn.Module):
    """Gated feed-forward network with a SiLU gate.

    SwiGLU beats a plain GELU MLP at equal parameter count in every transformer
    family it has been tried on, at the cost of a third weight matrix. The
    tensor-parallel plan column-shards ``gate_proj`` and ``up_proj`` and
    row-shards ``down_proj``, so the whole block needs exactly one all-reduce.

    Args:
        width: Input and output width.
        hidden: Inner width. Note that a SwiGLU with inner width ``h`` has
            ``3 * width * h`` parameters where a GELU MLP has ``2 * width * h``,
            so matching a 4x GELU MLP's parameter count means an inner width of
            ``8/3 * width``, not ``4 * width``. avgen states the inner width
            explicitly rather than hiding that factor in a ratio.
        bias: Whether to use biases. Off by default: biases add nothing
            measurable to a normalised transformer and complicate sharding.
    """

    def __init__(self, width: int, hidden: int, *, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden, bias=bias)
        self.up_proj = nn.Linear(width, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, width, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward network.

        The single positional argument is part of the tensor-parallel contract:
        ``standard_block_plan`` attaches a ``PrepareModuleInput`` that describes
        exactly one positional input here.

        Args:
            hidden: ``(batch, length, width)`` activations.

        Returns:
            ``(batch, length, width)`` activations.
        """
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))

    def init_weights(self, *, std: float = 0.02, out_std: float | None = None) -> None:
        """Initialise the three projections in place.

        Args:
            std: Standard deviation for the input projections.
            out_std: Standard deviation for the output projection, normally
                scaled down by depth so that the residual stream's variance does
                not grow linearly with the number of blocks.
        """
        init_linear(self.gate_proj, std=std)
        init_linear(self.up_proj, std=std)
        init_linear(self.down_proj, std=out_std if out_std is not None else std)


class SelfAttention(nn.Module):
    """Bidirectional self-attention with rotary positions and QK normalisation.

    Bidirectional, not causal: a diffusion transformer denoises the whole clip
    at once, so every token may attend to every other. That is also what makes
    context-parallel sharding trivially load-balanced — see
    :mod:`avgen.parallel.context`.

    Args:
        width: Model width.
        num_heads: Query heads.
        head_dim: Channels per head. Defaults to ``width // num_heads``.
        num_kv_heads: Key/value heads for grouped-query attention. Fewer
            key/value heads shrink the projections and, at inference, the cache;
            ``None`` means multi-head attention.
        qk_norm: Whether to RMS-normalise queries and keys per head.
        eps: Normalisation epsilon.
        bias: Whether the projections carry biases.

    Raises:
        ValueError: If the head counts do not divide, or ``width`` is not
            divisible by ``num_heads`` when ``head_dim`` is inferred.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if width % num_heads != 0:
                raise ValueError(
                    f"width {width} must be divisible by num_heads {num_heads} "
                    "when head_dim is not given explicitly"
                )
            head_dim = width // num_heads
        kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % kv_heads != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by num_kv_heads "
                f"{kv_heads} so every key/value head serves an equal group"
            )

        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(width, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(width, kv_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, width, bias=bias)
        self.q_norm = rms_norm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = rms_norm(head_dim, eps=eps) if qk_norm else nn.Identity()

    def _heads(self, projected: torch.Tensor) -> torch.Tensor:
        """Split a projection into heads, inferring the head count at runtime.

        Under column-wise tensor parallelism this module sees only its rank's
        slice of the heads, and the configured ``num_heads`` is the *global*
        count. The head dimension is invariant under that sharding, so dividing
        by it recovers the local head count at any parallel degree.
        """
        return projected.unflatten(-1, (-1, self.head_dim)).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        rope: RotaryTables | None = None,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over the sequence.

        The two positional parameters are part of the tensor-parallel contract:
        ``standard_block_plan`` declares a ``PrepareModuleInput`` with two
        positional inputs, the second untouched. ``key_mask`` is keyword-only so
        that it passes through the plan unmodified.

        Args:
            hidden: ``(batch, length, width)`` activations.
            rope: Rotary tables built from this rank's coordinate shard, or
                ``None`` for no positional encoding.
            key_mask: Precomputed SDPA mask from
                :func:`~avgen.models.attention.key_padding_mask`, or ``None``.

        Returns:
            ``(batch, length, width)`` activations.
        """
        query = self.q_norm(self._heads(self.q_proj(hidden)))
        key = self.k_norm(self._heads(self.k_proj(hidden)))
        value = self._heads(self.v_proj(hidden))

        if rope is not None:
            query = apply_rotary(query, rope)
            key = apply_rotary(key, rope)

        out = attention(
            query,
            key,
            value,
            attn_mask=key_mask,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return self.out_proj(out.transpose(1, 2).flatten(2))

    def init_weights(self, *, std: float = 0.02, out_std: float | None = None) -> None:
        """Initialise projections and QK norms in place.

        Args:
            std: Standard deviation for the q/k/v projections.
            out_std: Standard deviation for the output projection.
        """
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            init_linear(layer, std=std)
        init_linear(self.out_proj, std=out_std if out_std is not None else std)
        for norm in (self.q_norm, self.k_norm):
            init_norm(norm)


class CrossAttention(nn.Module):
    """Attention from the generative sequence into a frozen context.

    Used for two things: text conditioning, and — in :class:`AVDiT`'s
    ``cross_attention`` fusion — one modality attending into the other.

    No rotary embedding is applied. Text tokens have no physical time or space,
    so there is no coordinate to rotate by, and imposing an index-based rotation
    would teach the model that prompt word 7 sits "later" than prompt word 3 in
    the same phase space as video time. Cross-modal fusion instead expresses
    temporal locality through the mask, where it belongs.

    Args:
        width: Query-side model width.
        num_heads: Query heads.
        context_width: Key/value-side width. Defaults to ``width``; an
            asymmetric audio-video model uses a narrower one on one side.
        head_dim: Channels per head. Defaults to ``width // num_heads``.
        num_kv_heads: Key/value heads for grouped-query attention.
        qk_norm: Whether to RMS-normalise queries and keys per head.
        eps: Normalisation epsilon.
        bias: Whether the projections carry biases.

    Raises:
        ValueError: If the head counts do not divide.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        *,
        context_width: int | None = None,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if width % num_heads != 0:
                raise ValueError(
                    f"width {width} must be divisible by num_heads {num_heads} "
                    "when head_dim is not given explicitly"
                )
            head_dim = width // num_heads
        kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % kv_heads != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by num_kv_heads {kv_heads}"
            )
        keys_width = context_width if context_width is not None else width

        self.num_heads = num_heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(keys_width, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(keys_width, kv_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, width, bias=bias)
        self.q_norm = rms_norm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = rms_norm(head_dim, eps=eps) if qk_norm else nn.Identity()

    def _heads(self, projected: torch.Tensor) -> torch.Tensor:
        """Split a projection into heads, inferring the count at runtime."""
        return projected.unflatten(-1, (-1, self.head_dim)).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend from the sequence into the context.

        The three positional parameters are part of the tensor-parallel
        contract; ``standard_block_plan`` declares exactly this arity.

        Args:
            hidden: ``(batch, length, width)`` queries.
            context: ``(batch, context_length, context_width)`` keys and values.
            context_mask: Either a ``(batch, context_length)`` key-validity mask
                or a ready-made broadcastable SDPA mask of rank 4. The rank-4
                form is what temporally-local audio-video fusion uses, because
                its mask depends on the query as well as the key.

        Returns:
            ``(batch, length, width)`` activations.
        """
        if context.shape[1] == 0:
            # An empty context means unconditional generation. Returning zeros
            # keeps the residual untouched without a branch in the caller.
            return torch.zeros_like(hidden)

        mask = context_mask
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]

        query = self.q_norm(self._heads(self.q_proj(hidden)))
        key = self.k_norm(self._heads(self.k_proj(context)))
        value = self._heads(self.v_proj(context))
        out = attention(
            query,
            key,
            value,
            attn_mask=mask,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return self.out_proj(out.transpose(1, 2).flatten(2))

    def init_weights(self, *, std: float = 0.02, zero_output: bool = True) -> None:
        """Initialise projections and QK norms in place.

        Args:
            std: Standard deviation for the q/k/v projections.
            zero_output: Whether to zero the output projection. Cross-attention
                carries no adaLN gate, so zeroing the projection is what makes
                the branch start as the identity — and unlike zeroing a gated
                branch's projection it leaves the gradient path intact.
        """
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            init_linear(layer, std=std)
        init_linear(self.out_proj, std=std, zero=zero_output)
        for norm in (self.q_norm, self.k_norm):
            init_norm(norm)
