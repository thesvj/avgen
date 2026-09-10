"""Mixed-precision policy and optional float8 linear layers.

Three dtypes matter and they are not the same dtype:

``param_dtype``
    What weights are cast to for the forward and backward compute. ``bfloat16``
    for anything modern: it has the same exponent range as fp32, so it needs no
    loss scaling, which removes an entire class of "the run diverged at step
    40,000" failure that fp16 still suffers.

``reduce_dtype``
    What gradients are reduced in. **This one is worth being careful about.**
    Reducing in bf16 saves bandwidth but accumulates error across the reduction
    tree, and the tree is ``log(world_size)`` deep — so the error grows with the
    size of your job, which is exactly backwards from what you want. avgen
    defaults to fp32 reduction. On a 64-rank job the difference is invisible; on
    a 1024-rank job it is the difference between a clean loss curve and a slow
    upward drift nobody can explain.

``master_dtype``
    What the optimizer's copy of the weights lives in. fp32 always. bf16 has 8
    mantissa bits, and once ``lr * grad`` falls below the last bit of the weight
    the update is silently rounded to zero — late in training, when updates are
    small, this stalls learning entirely.

float8 is offered as an opt-in extra. It is a genuine ~1.3-1.5x speedup on
Hopper and Blackwell for large matmuls, and it is genuinely riskier: dynamic
scaling adds per-tensor reductions, and small or narrow layers get slower, not
faster. avgen applies it only to linear layers above a size threshold and never
to the input/output projections, where the numerics matter most and the compute
matters least.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

__all__ = [
    "PrecisionConfig",
    "convert_to_float8",
    "float8_available",
    "resolve_dtype",
]

DTypeName = Literal["float32", "bfloat16", "float16"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    """Resolve a dtype name to a ``torch.dtype``.

    Args:
        name: A dtype or one of its accepted spellings.

    Returns:
        The dtype.

    Raises:
        KeyError: If the name is not recognised.
    """
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[name.lower()]
    except KeyError as error:
        raise KeyError(
            f"unknown dtype {name!r}; supported: {', '.join(sorted(set(_DTYPES)))}"
        ) from error


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Numerical precision for compute, reduction, and the master weights.

    Args:
        param_dtype: Compute dtype for weights and activations.
        reduce_dtype: Dtype gradients are reduced in. Keep at float32.
        output_dtype: Dtype module outputs are cast to, or ``None`` to leave
            them in ``param_dtype``.
        enable_float8: Whether to swap eligible linear layers for float8.
        float8_min_features: Minimum in and out feature count for a linear layer
            to be converted. Below roughly this size the scaling overhead
            exceeds the matmul saving.
        float8_exclude: Module-name substrings never converted. Input and output
            projections stay in bf16 by default: they are a negligible fraction
            of the FLOPs and a large fraction of the numerical sensitivity.
        float8_recipe: Scaling strategy. ``tensorwise`` is the robust default;
            ``rowwise`` is more accurate and slightly slower.
    """

    param_dtype: DTypeName = "bfloat16"
    reduce_dtype: DTypeName = "float32"
    output_dtype: DTypeName | None = None
    enable_float8: bool = False
    float8_min_features: int = 1024
    float8_exclude: tuple[str, ...] = (
        "patch_embed",
        "proj_in",
        "proj_out",
        "final",
        "time_embed",
        "text_proj",
        "modulation",
    )
    float8_recipe: Literal["tensorwise", "rowwise"] = "tensorwise"
    _resolved: dict[str, torch.dtype] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate dtype names and the float8 threshold."""
        resolve_dtype(self.param_dtype)
        resolve_dtype(self.reduce_dtype)
        if self.output_dtype is not None:
            resolve_dtype(self.output_dtype)
        if isinstance(self.float8_min_features, bool) or self.float8_min_features < 1:
            raise ValueError(
                f"float8_min_features must be >= 1; got {self.float8_min_features!r}"
            )
        if self.param_dtype == "float16":
            # Not forbidden, but the caller should know what they signed up for.
            import warnings

            warnings.warn(
                "float16 parameters require loss scaling to avoid gradient "
                "underflow; bfloat16 is strongly preferred on any GPU that "
                "supports it (Ampere and later)",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def param(self) -> torch.dtype:
        """Resolved compute dtype."""
        return resolve_dtype(self.param_dtype)

    @property
    def reduce(self) -> torch.dtype:
        """Resolved gradient-reduction dtype."""
        return resolve_dtype(self.reduce_dtype)

    @property
    def output(self) -> torch.dtype | None:
        """Resolved output dtype, if pinned."""
        return None if self.output_dtype is None else resolve_dtype(self.output_dtype)

    def fsdp_policy(self) -> MixedPrecisionPolicy:
        """Return the FSDP2 mixed-precision policy this config describes.

        Returns:
            The policy to pass to ``fully_shard``.
        """
        return MixedPrecisionPolicy(
            param_dtype=self.param,
            reduce_dtype=self.reduce,
            output_dtype=self.output,
            cast_forward_inputs=True,
        )

    def autocast_dtype(self) -> torch.dtype | None:
        """Return the dtype for ``torch.autocast``, or ``None`` for fp32.

        Used only on the single-device path. Under FSDP2 the mixed-precision
        policy already casts parameters, and layering autocast on top would cast
        twice — harmless numerically, wasteful in practice.

        Returns:
            The autocast dtype, or ``None``.
        """
        return None if self.param is torch.float32 else self.param


def float8_available() -> bool:
    """Return whether float8 training is usable on this machine.

    Requires both ``torchao`` and a GPU with native float8 support (Hopper,
    sm_89 and later). Emulated float8 on older hardware is slower than bf16.

    Returns:
        Whether float8 can be enabled.
    """
    try:
        import torchao  # noqa: F401
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def _eligible(name: str, module: nn.Module, config: PrecisionConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if any(token in name for token in config.float8_exclude):
        return False
    if min(module.in_features, module.out_features) < config.float8_min_features:
        return False
    # float8 matmul kernels require both inner dimensions to be a multiple of
    # 16. A layer that is not gets silently padded or falls back, so exclude it
    # rather than pay for a conversion that buys nothing.
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def convert_to_float8(model: nn.Module, config: PrecisionConfig) -> int:
    """Swap eligible linear layers for float8 in place.

    Args:
        model: Model to convert.
        config: Precision configuration.

    Returns:
        The number of layers converted. Zero when float8 is disabled or
        unavailable, which is not an error — it lets the same config run on a
        Blackwell cluster and an Ampere workstation.

    Raises:
        RuntimeError: If float8 is explicitly enabled but ``torchao`` is not
            installed. Silently ignoring an explicit request would mean a user
            benchmarks bf16 and believes it is float8.
    """
    if not config.enable_float8:
        return 0
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as error:
        raise RuntimeError(
            "enable_float8=True requires torchao; install with "
            "`pip install 'avgen[quant]'` or set enable_float8=False"
        ) from error

    if not float8_available():
        import warnings

        warnings.warn(
            "float8 requested but this device lacks native float8 support "
            "(needs compute capability 8.9+); continuing in "
            f"{config.param_dtype}",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0

    eligible_names = {
        name
        for name, module in model.named_modules()
        if _eligible(name, module, config)
    }
    if not eligible_names:
        return 0

    float8_config = Float8LinearConfig.from_recipe_name(config.float8_recipe)
    convert_to_float8_training(
        model,
        config=float8_config,
        module_filter_fn=lambda _module, name: name in eligible_names,
    )
    return len(eligible_names)
