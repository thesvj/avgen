"""Side-tower and image-prompt adapters that leave the base model frozen.

Two adapters live here. Both add a *new* input modality to a model that was
never trained on it, and both start as an exact identity so the pretrained model
is not disturbed on step zero.

:class:`ControlAdapter` is ControlNet for a video DiT: a trainable copy of the
first N transformer blocks consumes an auxiliary control signal — a depth
sequence, a pose skeleton, an edge map, a low-resolution version of the target
video — and injects residuals into the frozen base at the matching block. The
copy starts from the base weights, so the side tower begins as a competent
feature extractor rather than as noise, and the injections are zero at
initialisation, so the base is untouched until the adapter has learned something
worth injecting.

:class:`IPAdapter` is decoupled cross-attention: an image prompt gets its own
key and value projections into the existing cross-attention, and its attention
output is added to the text branch's. Concatenating image tokens onto the text
sequence instead is the obvious alternative and it is worse — the text
projections were trained on text statistics, image embeddings drift them, and
the two conditioning strengths cannot then be balanced independently at
inference time. A separate branch keeps one scalar (``scale``) that trades image
adherence against prompt adherence, which is what users actually turn.

Attachment by hook, not by surgery
----------------------------------

Neither adapter rewrites the base model. Both attach ``forward`` hooks, because
a hook needs to know only that a block *returns a tensor* — it never has to know
the block's call signature, how many extra conditioning arguments it takes, or
what order they are in. That keeps the adapters working across every model in
:mod:`avgen.models` and across a user's own model, and it means detaching is
exact: remove the handles and the frozen base is byte-for-byte what it was.

The only structural assumptions are the ones the build contract already
guarantees: a model exposes ``blocks`` as an ``nn.ModuleList``, and a block's
cross-attention submodule is named ``cross_attention`` with a ``q_proj`` inside.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.hooks import RemovableHandle

__all__ = [
    "AdapterHandles",
    "ControlAdapter",
    "ControlAdapterConfig",
    "IPAdapter",
    "IPAdapterConfig",
]


class AdapterHandles:
    """A removable group of module hooks.

    Usable as a context manager so a temporary attachment — an evaluation pass
    with control, a debugging comparison against the bare base — cannot leak a
    hook into the rest of the run.

    Args:
        handles: The hook handles to own.
    """

    __slots__ = ("_handles",)

    def __init__(self, handles: Sequence[RemovableHandle]) -> None:
        self._handles = list(handles)

    def __len__(self) -> int:
        """Number of live hooks."""
        return len(self._handles)

    def remove(self) -> None:
        """Remove every hook. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> AdapterHandles:
        """Return self so ``with adapter.attach(model) as handles`` reads well."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove the hooks on scope exit, including on an exception."""
        self.remove()


def _blocks_of(model: nn.Module, attribute: str) -> nn.ModuleList:
    """Return a model's block list.

    Args:
        model: The model to inspect.
        attribute: Attribute name holding the blocks.

    Returns:
        The block list.

    Raises:
        AttributeError: If the attribute is absent or not an ``nn.ModuleList``.
    """
    blocks = getattr(model, attribute, None)
    if not isinstance(blocks, nn.ModuleList):
        raise AttributeError(
            f"model {type(model).__name__} has no nn.ModuleList attribute "
            f"{attribute!r}; the build contract requires one so that adapters, "
            "FSDP and activation checkpointing can all address blocks uniformly"
        )
    return blocks


def _split_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...]]:
    """Split a block's return value into its hidden state and any extras.

    Blocks that return a bare tensor and blocks that return
    ``(hidden, *auxiliaries)`` are both common; supporting both here is what
    keeps the adapters signature-agnostic.

    Args:
        output: A block's return value.

    Returns:
        ``(hidden_state, extras)``.

    Raises:
        TypeError: If the leading element is not a tensor.
    """
    if isinstance(output, torch.Tensor):
        return output, ()
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], tuple(output[1:])
    raise TypeError(
        "adapter injection needs a block whose output is a tensor or a tuple "
        f"whose first element is a tensor; got {type(output).__name__}"
    )


def _rejoin(hidden: torch.Tensor, extras: tuple[Any, ...], original: Any) -> Any:
    """Rebuild a block output of the original shape with a new hidden state."""
    if not extras and isinstance(original, torch.Tensor):
        return hidden
    return (hidden, *extras)


def _zero_linear(in_features: int, out_features: int) -> nn.Linear:
    """Return a linear whose weight and bias are exactly zero.

    ControlNet's "zero convolution", in token space. Zero output means the
    residual injected into the frozen base is exactly zero at initialisation, so
    the adapted model reproduces the base model bit-for-bit before the first
    optimizer step. The gradient with respect to this layer's weight is not zero
    (it is the incoming gradient times the side-tower activation), so the layer
    learns immediately — the zero is an initial condition, not a dead branch.
    """
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


@dataclass(frozen=True, slots=True)
class ControlAdapterConfig:
    """How the control side tower is built and attached.

    Args:
        control_width: Feature width of one control token, as produced by
            whatever encodes the control signal (a patchified depth latent, a
            pose heatmap, a downsampled video latent).
        num_blocks: How many leading base blocks the tower copies. ControlNet
            copies the encoder half; for an isotropic DiT with no encoder/decoder
            split, the leading third to half of the depth is the usual choice.
            More blocks is more control authority and linearly more memory.
        model_width: Hidden width of the base model. ``None`` infers it from the
            captured block input at attach time, which is the honest default
            because the model config is not something this module can see.
        block_attribute: Attribute holding the base ``nn.ModuleList``.
        conditioning_scale: Multiplier on every injected residual. The knob a
            user turns at inference to trade control strength against prompt
            adherence; 1.0 during training.
        injection_stride: Inject into every ``stride``-th block rather than
            every one. Halves the number of zero projections and, in practice,
            costs very little control fidelity.

    Raises:
        ValueError: If any dimension or count is non-positive.
    """

    control_width: int
    num_blocks: int = 6
    model_width: int | None = None
    block_attribute: str = "blocks"
    conditioning_scale: float = 1.0
    injection_stride: int = 1

    def __post_init__(self) -> None:
        """Validate widths and counts.

        Raises:
            ValueError: On a non-positive field.
        """
        for name in ("control_width", "num_blocks", "injection_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.model_width is not None and self.model_width < 1:
            raise ValueError(
                "model_width must be a positive integer or None; got "
                f"{self.model_width!r}"
            )
        if self.conditioning_scale < 0.0:
            raise ValueError(
                f"conditioning_scale must be non-negative; got "
                f"{self.conditioning_scale!r}"
            )


class ControlAdapter(nn.Module):
    """A ControlNet-style trainable side tower over a frozen base model.

    The tower is a deep copy of the base model's first ``num_blocks`` blocks.
    Copying rather than initialising fresh is the entire trick from ControlNet:
    a randomly initialised side tower has to learn video features from scratch
    with a frozen teacher's gradient as its only signal, which takes far longer
    than the control task itself; a copy already knows what a video looks like
    and only has to learn what the control signal means.

    The rejected alternative is fine-tuning the base with the control signal
    concatenated to the input. That works and is simpler, but it destroys the
    base model — you get one model per control modality, none of which can be
    composed, and the original weights are gone. A side tower is additive: depth
    control and pose control are two adapter files over one shared base, and
    they can be attached together.

    Data flow per forward pass, all driven by hooks:

    1. A pre-hook on ``blocks[0]`` captures the hidden state entering the base
       stack, together with whatever extra arguments the block was called with.
    2. The tower runs on ``hidden + control_embed(control_tokens)`` using those
       same extra arguments, so the tower sees exactly the conditioning the base
       sees — timestep embedding, text context, rotary tables — with no
       knowledge of what they are.
    3. Each tower block's output goes through its own zero-initialised
       projection and is added to the corresponding base block's output.

    Args:
        base: The frozen base model. Not stored as a submodule — see
            :meth:`attach` — only read from at construction time.
        config: Tower geometry.
    """

    def __init__(self, base: nn.Module, config: ControlAdapterConfig) -> None:
        super().__init__()
        self.config = config
        base_blocks = _blocks_of(base, config.block_attribute)
        if config.num_blocks > len(base_blocks):
            raise ValueError(
                f"num_blocks={config.num_blocks} exceeds the base model's "
                f"{len(base_blocks)} blocks"
            )
        # A deep copy, then unfreeze: the tower is trainable even though the
        # module it was copied from is not, and copy.deepcopy carries
        # requires_grad across.
        self.blocks = nn.ModuleList(
            copy.deepcopy(block) for block in base_blocks[: config.num_blocks]
        )
        for parameter in self.blocks.parameters():
            parameter.requires_grad_(True)
        self.injection_indices: tuple[int, ...] = tuple(
            range(0, config.num_blocks, config.injection_stride)
        )
        self.control_embed: nn.Linear | None = None
        self.projections: nn.ModuleList | None = None
        if config.model_width is not None:
            self._build_projections(config.model_width)

    def _build_projections(self, model_width: int) -> tuple[nn.Linear, nn.ModuleList]:
        """Create the control input embedding and the zero output projections.

        Args:
            model_width: Hidden width of the base model.

        Returns:
            The control embedding and the zero-initialised output projections.
        """
        embed = nn.Linear(self.config.control_width, model_width)
        # Small but non-zero: the control signal must actually reach the tower.
        # Only the *output* projections need to be zero for exact identity, and
        # zeroing the input as well would make the tower's first gradient zero
        # too, delaying learning by however long it takes the output projection
        # to move off zero.
        nn.init.normal_(embed.weight, std=1.0 / math.sqrt(self.config.control_width))
        nn.init.zeros_(embed.bias)
        projections = nn.ModuleList(
            _zero_linear(model_width, model_width) for _ in self.injection_indices
        )
        self.control_embed = embed
        self.projections = projections
        return embed, projections

    def _ensure_built(self, hidden: torch.Tensor) -> tuple[nn.Linear, nn.ModuleList]:
        """Build the projections on first use, once the model width is known.

        The width is read from the hidden state rather than required up front,
        because the base model's config is not something this module can see and
        demanding it duplicates a number that would then be able to disagree.

        Args:
            hidden: The hidden state entering the base stack.

        Returns:
            The control embedding and the output projections.
        """
        embed, projections = self.control_embed, self.projections
        if embed is None or projections is None:
            embed, projections = self._build_projections(int(hidden.shape[-1]))
            embed.to(device=hidden.device, dtype=hidden.dtype)
            projections.to(device=hidden.device, dtype=hidden.dtype)
        return embed, projections

    def forward(
        self,
        hidden: torch.Tensor,
        control: torch.Tensor,
        *block_args: Any,
        **block_kwargs: Any,
    ) -> list[torch.Tensor]:
        """Run the tower and return one residual per injection point.

        Args:
            hidden: ``(batch, length, width)`` hidden state entering the base
                stack.
            control: ``(batch, length, control_width)`` control tokens, already
                aligned one-to-one with the base tokens. Alignment is the
                caller's job because only the caller knows the control
                modality's own patchification.
            *block_args: Extra positional arguments the base block was called
                with, forwarded verbatim.
            **block_kwargs: Extra keyword arguments, forwarded verbatim.

        Returns:
            Residuals in injection order.

        Raises:
            ValueError: If the control tokens do not align with the hidden
                state in batch or length.
        """
        control_embed, projections = self._ensure_built(hidden)
        if control.shape[:2] != hidden.shape[:2]:
            raise ValueError(
                f"control tokens must align with the base sequence "
                f"{tuple(hidden.shape[:2])}; got {tuple(control.shape[:2])}"
            )
        state = hidden + control_embed(control.to(hidden.dtype))
        residuals: list[torch.Tensor] = []
        projection_index = 0
        for index, block in enumerate(self.blocks):
            state, _ = _split_output(block(state, *block_args, **block_kwargs))
            if index in self.injection_indices:
                projection = projections[projection_index]
                residuals.append(projection(state) * self.config.conditioning_scale)
                projection_index += 1
        return residuals

    def attach(self, base: nn.Module, control: torch.Tensor) -> AdapterHandles:
        """Hook the tower onto a base model for one control signal.

        The base model is not stored on this module and this module is not
        stored on the base. Keeping them separate means the base's state dict
        never gains adapter keys, the adapter's state dict never gains a
        duplicate of the base, and ``parallelize`` can shard either one without
        accidentally reaching the other through a back-reference.

        Args:
            base: The frozen base model.
            control: ``(batch, length, control_width)`` control tokens for the
                forward pass about to run.

        Returns:
            Handles to remove when the pass is done. Use as a context manager.
        """
        blocks = _blocks_of(base, self.config.block_attribute)
        residuals: dict[int, torch.Tensor] = {}
        handles: list[RemovableHandle] = []

        def _capture(
            _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            hidden = args[0]
            computed = self(hidden, control, *args[1:], **kwargs)
            residuals.clear()
            residuals.update(dict(zip(self.injection_indices, computed, strict=True)))

        handles.append(blocks[0].register_forward_pre_hook(_capture, with_kwargs=True))

        def _make_injector(index: int) -> Any:
            def _inject(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
                residual = residuals.get(index)
                if residual is None:
                    return output
                hidden, extras = _split_output(output)
                return _rejoin(hidden + residual.to(hidden.dtype), extras, output)

            return _inject

        for index in self.injection_indices:
            handles.append(blocks[index].register_forward_hook(_make_injector(index)))
        return AdapterHandles(handles)


@dataclass(frozen=True, slots=True)
class IPAdapterConfig:
    """How image-prompt cross-attention is injected.

    Args:
        image_width: Feature width of the image-prompt embedding, as produced by
            a frozen image encoder.
        num_tokens: How many tokens the image prompt is projected to. Four is
            the original IP-Adapter setting; sixteen is what the "plus" variant
            uses to carry composition rather than only style.
        scale: Strength of the image branch, added to the text branch's output.
            Exposed at inference and swept there; 1.0 during training.
        target_blocks: Patterns selecting which blocks receive the injection.
            ``()`` means every block that has a ``cross_attention`` submodule.
        num_heads: Attention head count. ``None`` reads ``num_heads`` from the
            target attention module, which every model in this framework
            exposes; pass it explicitly for a model that does not.
        attention_attribute: Name of the cross-attention submodule inside a
            block, per the model contract.

    Raises:
        ValueError: If a dimension or count is non-positive.
    """

    image_width: int
    num_tokens: int = 4
    scale: float = 1.0
    target_blocks: tuple[str, ...] = ()
    num_heads: int | None = None
    attention_attribute: str = "cross_attention"

    def __post_init__(self) -> None:
        """Validate widths and counts.

        Raises:
            ValueError: On a non-positive field.
        """
        for name in ("image_width", "num_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.num_heads is not None and self.num_heads < 1:
            raise ValueError(
                f"num_heads must be a positive integer or None; got {self.num_heads!r}"
            )
        if self.scale < 0.0:
            raise ValueError(f"scale must be non-negative; got {self.scale!r}")


class IPAdapter(nn.Module):
    """Decoupled image-prompt cross-attention over a frozen base.

    For each targeted block, the image prompt gets its own ``k`` and ``v``
    projections. The queries are the base model's own — read by calling the
    frozen ``q_proj`` on the same hidden state the block's cross-attention
    received — so the image branch attends from exactly the same query space as
    the text branch, and the two outputs are commensurable enough to add.

    The output projection is zero-initialised, so at initialisation the image
    branch contributes exactly nothing and the base model is untouched.

    Args:
        base: The frozen base model, read at construction to find widths.
        config: Injection settings.
    """

    def __init__(self, base: nn.Module, config: IPAdapterConfig) -> None:
        super().__init__()
        self.config = config
        targets = self._find_targets(base)
        if not targets:
            raise ValueError(
                "no block exposes a cross-attention submodule named "
                f"{config.attention_attribute!r}; an IP-adapter needs one to "
                "borrow queries from"
            )
        self.target_names: tuple[str, ...] = tuple(name for name, _ in targets)
        widths = {int(module.q_proj.out_features) for _, module in targets}
        if len(widths) != 1:
            raise ValueError(
                f"targeted cross-attention modules disagree on width: {sorted(widths)}"
            )
        width = widths.pop()
        self.width = width
        # One resampler for the whole model: the image prompt is the same at
        # every depth, so projecting it per block would be num_blocks copies of
        # the same computation and the same parameters.
        self.image_proj = nn.Linear(config.image_width, width * config.num_tokens)
        nn.init.normal_(self.image_proj.weight, std=1.0 / math.sqrt(config.image_width))
        nn.init.zeros_(self.image_proj.bias)
        self.image_norm = nn.LayerNorm(width)
        self.k_proj = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in targets
        )
        self.v_proj = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in targets
        )
        self.out_proj = nn.ModuleList(_zero_linear(width, width) for _ in targets)
        self.heads = self._resolve_heads(targets)

    def _find_targets(self, base: nn.Module) -> list[tuple[str, nn.Module]]:
        """Return the cross-attention modules to inject into, in walk order."""
        from avgen.finetune._match import matches_any

        found: list[tuple[str, nn.Module]] = []
        for name, module in base.named_modules():
            if name.rpartition(".")[2] != self.config.attention_attribute:
                continue
            if not hasattr(module, "q_proj"):
                continue
            if self.config.target_blocks and not matches_any(
                name, self.config.target_blocks
            ):
                continue
            found.append((name, module))
        return found

    def _resolve_heads(self, targets: Sequence[tuple[str, nn.Module]]) -> int:
        """Determine the head count for the image branch.

        Args:
            targets: The targeted attention modules.

        Returns:
            Number of attention heads.

        Raises:
            ValueError: If the count cannot be determined.
        """
        if self.config.num_heads is not None:
            return self.config.num_heads
        for _, module in targets:
            heads = getattr(module, "num_heads", None)
            if isinstance(heads, int) and heads > 0:
                return heads
        raise ValueError(
            "cannot infer num_heads from the target attention modules; pass "
            "IPAdapterConfig(num_heads=...) explicitly"
        )

    def image_tokens(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """Project an image embedding into cross-attention context tokens.

        Args:
            image_embedding: ``(batch, image_width)`` pooled embedding, or
                ``(batch, tokens, image_width)`` which is mean-pooled first.

        Returns:
            ``(batch, num_tokens, width)`` context tokens.
        """
        pooled = (
            image_embedding.mean(dim=1)
            if image_embedding.ndim == 3
            else image_embedding
        )
        projected = self.image_proj(pooled)
        tokens = projected.reshape(pooled.shape[0], self.config.num_tokens, self.width)
        return self.image_norm(tokens)

    def _branch(
        self,
        index: int,
        attention: nn.Module,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the image branch's contribution for one block.

        Args:
            index: Position of this block among the targets.
            attention: The frozen cross-attention module, used for its queries.
            hidden: ``(batch, length, width)`` input to the cross-attention.
            context: ``(batch, num_tokens, width)`` image context.

        Returns:
            ``(batch, length, width)`` residual to add to the block's output.
        """
        batch, length, _ = hidden.shape
        heads = self.heads
        head_dim = self.width // heads

        def _split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(tensor.shape[0], -1, heads, head_dim).transpose(1, 2)

        query = _split(attention.q_proj(hidden))
        key = _split(self.k_proj[index](context))
        value = _split(self.v_proj[index](context))
        attended = functional.scaled_dot_product_attention(query, key, value)
        merged = attended.transpose(1, 2).reshape(batch, length, self.width)
        return self.out_proj[index](merged) * self.config.scale

    def attach(self, base: nn.Module, image_embedding: torch.Tensor) -> AdapterHandles:
        """Hook the image branch onto a base model for one image prompt.

        Args:
            base: The frozen base model.
            image_embedding: The image prompt, in either shape accepted by
                :meth:`image_tokens`.

        Returns:
            Handles to remove when the pass is done.
        """
        context = self.image_tokens(image_embedding)
        handles: list[RemovableHandle] = []
        pending: dict[int, torch.Tensor] = {}

        def _make_capture(index: int) -> Any:
            def _capture(
                _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> None:
                hidden = args[0] if args else kwargs["hidden_states"]
                pending[index] = hidden

            return _capture

        def _make_inject(index: int, attention: nn.Module) -> Any:
            def _inject(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
                hidden = pending.pop(index, None)
                if hidden is None:
                    return output
                base_out, extras = _split_output(output)
                residual = self._branch(index, attention, hidden, context)
                return _rejoin(base_out + residual.to(base_out.dtype), extras, output)

            return _inject

        for index, name in enumerate(self.target_names):
            attention = base.get_submodule(name)
            handles.append(
                attention.register_forward_pre_hook(
                    _make_capture(index), with_kwargs=True
                )
            )
            handles.append(
                attention.register_forward_hook(_make_inject(index, attention))
            )
        return AdapterHandles(handles)
