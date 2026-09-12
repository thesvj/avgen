"""Optimizer construction, and the parameter split that most code gets wrong.

There is one non-obvious decision in this module and it is worth the space:
**weight decay must not be applied to normalisation gains, biases, modulation
tables, or embeddings.**

For a weight matrix, decay is a genuine capacity constraint: it penalises the
function the layer computes. For a normalisation gain it is not, because the
layer immediately after can undo any rescaling. Decaying a LayerNorm gain
towards zero shrinks that layer's output scale, the next layer's weights grow to
compensate, and the network ends up with the same function, a worse condition
number, and a gradient scale that drifts over training. You paid for
regularisation and bought numerical instability.

In a diffusion transformer there is a sharper version of the same mistake. The
timestep and text conditioning reach every block through an adaptive-norm
``scale_shift`` table that is deliberately zero-initialised, so that each block
starts as the identity and the residual stream is undisturbed at step zero.
Weight decay pulls exactly that table back towards zero — the state it was
designed to *leave*. The symptom is a model that trains, produces plausible
images, and ignores its conditioning: the path carrying ``t`` and the prompt was
regularised out of existence.

Embeddings are excluded for a related reason: a token seen rarely gets decayed
on every step but updated on almost none, so decay acts as a frequency-dependent
penalty that has nothing to do with generalisation.

The rule implemented here is name patterns plus a dimensionality fallback —
anything with one dimension or fewer is a gain, a bias, or a scalar, and none of
those should be decayed regardless of what it is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = [
    "DEFAULT_NO_DECAY_PATTERNS",
    "build_optimizer",
    "build_param_groups",
    "list_optimizers",
]

#: Substrings matched case-insensitively against the parameter's qualified name.
#: Substrings rather than regexes because the names they must catch are fixed by
#: the model contract (``attention_norm``, ``ffn_norm``, ``scale_shift``,
#: ``patch_embed``, ``time_embed``) and a regex here would invite the kind of
#: cleverness that silently stops matching when a submodule is renamed.
DEFAULT_NO_DECAY_PATTERNS: tuple[str, ...] = ("norm", "bias", "scale_shift", "embed")

_OPTIMIZERS: tuple[str, ...] = ("adafactor", "adamw", "adamw_8bit")


def list_optimizers() -> tuple[str, ...]:
    """Return every supported optimizer name, sorted.

    Returns:
        Sorted optimizer names.
    """
    return _OPTIMIZERS


def build_param_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    no_decay_patterns: Sequence[str] = DEFAULT_NO_DECAY_PATTERNS,
) -> list[dict[str, Any]]:
    """Split trainable parameters into decayed and undecayed groups.

    Iteration follows ``named_parameters`` order, which is the module
    registration order and therefore identical on every rank. That matters more
    than it looks: optimizer state is checkpointed per group, so two ranks
    disagreeing about group membership produces a checkpoint that loads without
    error and restores the wrong moments.

    Args:
        model: The module whose parameters to group.
        weight_decay: Decay applied to the decayed group.
        no_decay_patterns: Case-insensitive substrings marking a parameter as
            undecayed.

    Returns:
        Two parameter-group dictionaries, decayed first. A group with no members
        is omitted, because some optimizers reject an empty group.

    Raises:
        ValueError: If the model has no trainable parameters, which is almost
            always a frozen-module bug rather than an intention.
    """
    patterns = tuple(pattern.lower() for pattern in no_decay_patterns)
    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        # The dimensionality test is the backstop: a rank-1 tensor is a gain, a
        # bias, or a scalar under any naming convention, and none of them should
        # be decayed even in a model that names them something unexpected.
        if parameter.ndim <= 1 or any(pattern in lowered for pattern in patterns):
            undecayed.append(parameter)
        else:
            decayed.append(parameter)
    if not decayed and not undecayed:
        raise ValueError(
            "model has no trainable parameters; check that a freeze helper did "
            "not disable requires_grad on everything"
        )
    groups: list[dict[str, Any]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


def _should_fuse(groups: Sequence[dict[str, Any]], fused: bool | None) -> bool:
    """Decide whether to request the fused optimizer kernel.

    Fused AdamW folds the whole update into one kernel per dtype, which removes
    a few thousand tiny launches per step. On a large model that is a few
    percent of step time; on a small one at high step rate it is much more.

    Args:
        groups: The parameter groups.
        fused: Explicit request, or ``None`` to decide automatically.

    Returns:
        Whether to pass ``fused=True``.
    """
    if fused is not None:
        return fused
    # Deliberately guarded and inside a function: importing avgen must not touch
    # the CUDA driver, and the fused path is CUDA-only.
    if not torch.cuda.is_available():
        return False
    return all(
        parameter.device.type == "cuda"
        for group in groups
        for parameter in group["params"]
    )


def build_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = None,
    no_decay_patterns: Sequence[str] = DEFAULT_NO_DECAY_PATTERNS,
) -> Optimizer:
    """Build an optimizer with correctly split parameter groups.

    ``betas`` defaults to ``(0.9, 0.95)`` rather than the PyTorch default
    ``(0.9, 0.999)``. A second-moment horizon of a thousand steps adapts far too
    slowly for a diffusion objective, whose per-step gradient scale varies with
    the sampled noise level; the shorter horizon is what every large transformer
    recipe converged on.

    Args:
        model: The module to optimize.
        name: ``"adamw"``, ``"adamw_8bit"``, or ``"adafactor"``.
        lr: Peak learning rate. A schedule multiplies this.
        weight_decay: Decay for the decayed group only.
        betas: Adam moment decay rates.
        eps: Denominator epsilon.
        fused: Force the fused kernel on or off; ``None`` auto-detects CUDA.
        no_decay_patterns: Substrings marking a parameter as undecayed.

    Returns:
        The constructed optimizer.

    Raises:
        ValueError: If a hyper-parameter is out of range or ``name`` is unknown.
        RuntimeError: If an optional backend is requested but not installed.
    """
    if lr <= 0.0:
        raise ValueError(f"lr must be positive; got {lr!r}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative; got {weight_decay!r}")
    if not all(0.0 <= beta < 1.0 for beta in betas):
        raise ValueError(f"betas must lie in [0, 1); got {betas!r}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps!r}")

    groups = build_param_groups(
        model, weight_decay=weight_decay, no_decay_patterns=no_decay_patterns
    )
    if name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=lr,
            betas=betas,
            eps=eps,
            fused=_should_fuse(groups, fused),
        )
    if name == "adamw_8bit":
        return _build_adamw_8bit(groups, lr=lr, betas=betas, eps=eps)
    if name == "adafactor":
        # Adafactor factorises the second moment into row and column statistics,
        # so its state is O(n + m) instead of O(n * m). That is a real saving
        # only when the optimizer state dominates, which under FSDP2 sharding it
        # usually does not — prefer AdamW unless memory has actually forced the
        # issue, because the factorisation costs convergence quality.
        return torch.optim.Adafactor(
            groups,
            lr=lr,
            beta2_decay=-0.8,
            eps=(None, eps),
            weight_decay=weight_decay,
        )
    available = ", ".join(list_optimizers())
    raise ValueError(f"unknown optimizer {name!r}; available: {available}")


def _build_adamw_8bit(
    groups: Sequence[dict[str, Any]],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
) -> Optimizer:
    """Build bitsandbytes' 8-bit AdamW, imported lazily.

    Blockwise 8-bit quantisation of the two Adam moments cuts optimizer state
    from 8 bytes per parameter to 2. Note what that does *not* compose with:
    the quantised state is a plain local tensor, so it does not survive DTensor
    sharding, which rules it out under FSDP2 or tensor parallelism. It is the
    right tool for single-GPU fine-tuning and the wrong one for a cluster run.

    Args:
        groups: Parameter groups.
        lr: Peak learning rate.
        betas: Adam moment decay rates.
        eps: Denominator epsilon.

    Returns:
        The 8-bit optimizer.

    Raises:
        RuntimeError: If bitsandbytes is not installed.
    """
    try:
        import bitsandbytes
    except ImportError as error:
        raise RuntimeError(
            "the 'adamw_8bit' optimizer requires bitsandbytes; "
            "install it with `pip install bitsandbytes`"
        ) from error
    return bitsandbytes.optim.AdamW8bit(list(groups), lr=lr, betas=betas, eps=eps)
