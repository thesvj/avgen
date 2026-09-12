"""Activation checkpointing policies.

Activation memory, not parameter memory, is what stops a video model from
fitting. A 2B-parameter model needs ~4 GB of bf16 weights; the activations for a
single 100k-token sequence through 40 blocks can be many times that. So the
checkpointing policy is a first-class training knob, not an afterthought.

Four policies, in increasing order of memory saved and compute spent:

``none``
    Store everything. Correct choice for small models and short sequences,
    where the extra forward pass is pure loss.

``selective_op``
    **The default, and the one that is right most of the time.** Store the
    outputs of operations that are expensive to recompute — matmuls, SDPA — and
    recompute everything that is cheap — normalisation, activation functions,
    elementwise arithmetic. Typically recovers most of the memory of full
    checkpointing for a small fraction of the compute, because in a transformer
    the cheap ops dominate the *count* of saved tensors while the expensive ops
    dominate their *size*.

``selective_layer``
    Checkpoint every n-th block entirely. Coarser and usually worse than
    ``selective_op``, but predictable, and it composes cleanly with pipeline
    parallelism where per-stage memory must be balanced by hand.

``full``
    Checkpoint every block. Maximum memory saving, roughly 30-40% more compute.
    Reach for it when the sequence is long enough that nothing else fits.

The op list below is the load-bearing part. It is expressed against
``torch.ops.aten`` because that is the level ``torch.utils.checkpoint``'s
policy function actually sees, and because it then applies unchanged to any
model built from standard PyTorch operations — including one a user writes
themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

__all__ = [
    "ACMode",
    "ActivationCheckpointConfig",
    "apply_activation_checkpointing",
]

ACMode = Literal["none", "full", "selective_op", "selective_layer"]

_aten = torch.ops.aten

#: Operations whose outputs are saved rather than recomputed.
#:
#: Chosen on one criterion: is recomputing this materially more expensive than
#: storing it? Matmuls and attention are; norms and activations are not. The
#: scaled-dot-product variants are listed individually rather than by prefix
#: because the flash and memory-efficient kernels appear under distinct op
#: names and missing one silently loses most of the benefit.
DEFAULT_SAVE_OPS: tuple[Any, ...] = (
    _aten.mm.default,
    _aten.addmm.default,
    _aten.bmm.default,
    _aten._scaled_dot_product_efficient_attention.default,
    _aten._scaled_dot_product_flash_attention.default,
    _aten._scaled_mm.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_gather_into_tensor.default,
)


@dataclass(frozen=True, slots=True)
class ActivationCheckpointConfig:
    """How aggressively to trade compute for activation memory.

    Args:
        mode: Which policy to apply.
        layer_interval: For ``selective_layer``, checkpoint one block in every
            ``layer_interval``. ``1`` is equivalent to ``full``.
        save_op_frequency: For ``selective_op``, save the output of every
            n-th occurrence of a listed op and recompute the rest. ``1`` saves
            all of them. Raising it to ``2`` or ``3`` gives a finer memory
            dial than switching whole modes, which matters when a
            configuration is a few hundred megabytes from fitting.
        early_stop: Whether recomputation may stop as soon as the needed
            tensors have been produced, rather than always replaying a whole
            block. Pure win; off only for debugging recomputation itself.

    Raises:
        ValueError: If an interval or frequency is not positive.
    """

    mode: ACMode = "selective_op"
    layer_interval: int = 2
    save_op_frequency: int = 1
    early_stop: bool = True

    def __post_init__(self) -> None:
        """Validate the interval and frequency."""
        for name in ("layer_interval", "save_op_frequency"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be >= 1; got {value!r}")
        if self.mode not in ("none", "full", "selective_op", "selective_layer"):
            raise ValueError(f"unsupported activation checkpoint mode {self.mode!r}")

    @property
    def enabled(self) -> bool:
        """Whether any checkpointing is applied."""
        return self.mode != "none"


def _selective_op_context(config: ActivationCheckpointConfig) -> Callable[..., Any]:
    """Build the context factory implementing the selective-op policy."""
    save_ops = set(DEFAULT_SAVE_OPS)
    frequency = config.save_op_frequency

    def policy_factory() -> Callable[..., CheckpointPolicy]:
        counts: dict[Any, int] = {}

        def policy(
            _context: Any,
            func: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> CheckpointPolicy:
            if func not in save_ops:
                return CheckpointPolicy.PREFER_RECOMPUTE
            counts[func] = counts.get(func, 0) + 1
            # Save every n-th occurrence; recompute the others. Counting per-op
            # rather than globally keeps the saved set balanced across the
            # different matmuls in a block instead of favouring whichever runs
            # first.
            should_save = (counts[func] - 1) % frequency == 0
            return (
                CheckpointPolicy.MUST_SAVE
                if should_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return policy

    def context_fn() -> tuple[Any, Any]:
        return cast(
            "tuple[Any, Any]", create_selective_checkpoint_contexts(policy_factory())
        )

    return context_fn


def _wrap(module: nn.Module, config: ActivationCheckpointConfig) -> nn.Module:
    """Wrap one module according to the configured policy."""
    if config.mode == "selective_op":
        return cast(
            "nn.Module",
            checkpoint_wrapper(
                module,
                context_fn=_selective_op_context(config),
                preserve_rng_state=False,
                early_stop=config.early_stop,
            ),
        )
    return cast(
        "nn.Module",
        checkpoint_wrapper(
            module,
            preserve_rng_state=False,
            early_stop=config.early_stop,
        ),
    )


def apply_activation_checkpointing(
    model: nn.Module,
    config: ActivationCheckpointConfig,
    *,
    block_attribute: str = "blocks",
) -> nn.Module:
    """Apply the configured checkpointing policy to a model's transformer blocks.

    Blocks are replaced inside their parent ``ModuleList`` in place, preserving
    their names. That matters: FSDP2 wrapping, tensor-parallel plans, and
    checkpoint state-dict keys are all addressed by module path, so a wrapper
    that renamed ``blocks.0`` to ``blocks.0._checkpoint_wrapped_module`` would
    break all three at once.

    ``preserve_rng_state`` is deliberately false. Recomputation must be
    deterministic, which means blocks must not contain dropout or any other
    randomness — a constraint avgen's models satisfy, because diffusion
    transformers do not use dropout in the residual stream. Preserving RNG state
    would otherwise cost a device synchronisation per block per step.

    Args:
        model: The model to modify in place.
        config: The policy to apply.
        block_attribute: Name of the ``ModuleList`` holding transformer blocks.

    Returns:
        The same model, modified in place.

    Raises:
        AttributeError: If the model has no such attribute.
        TypeError: If the attribute is not a ``ModuleList``.
    """
    if not config.enabled:
        return model

    blocks = getattr(model, block_attribute, None)
    if blocks is None:
        raise AttributeError(
            f"model {type(model).__name__} has no {block_attribute!r} attribute; "
            "pass block_attribute= to point at its transformer blocks"
        )
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(
            f"{block_attribute!r} must be an nn.ModuleList; got {type(blocks).__name__}"
        )

    for index, block in enumerate(blocks):
        if config.mode == "selective_layer" and index % config.layer_interval != 0:
            continue
        blocks[index] = _wrap(block, config)
    return model
