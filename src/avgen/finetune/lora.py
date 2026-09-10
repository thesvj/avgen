"""LoRA and DoRA, implemented natively against sharded parameters.

Low-rank adaptation replaces a weight update ``ΔW ∈ R^{outxin}`` by a rank-``r``
factorisation ``ΔW = (alpha/r)·B·A`` with ``B ∈ R^{outxr}`` and ``A ∈ R^{rxin}``.
For a video DiT the practical consequence is not "fewer parameters" — it is that
the optimizer state, which is two fp32 moments per trainable parameter and is
usually the single largest allocation in a training job, shrinks by three orders
of magnitude. A 2B-parameter model fine-tunes on one 80GB GPU instead of eight.

**DoRA** (weight-decomposed low-rank adaptation) splits the adapted weight into a
direction and a per-output-unit magnitude::

    W' = m ⊙ (W + s·B·A) / ‖W + s·B·A‖_row

Full fine-tuning changes both the direction and the magnitude of each output
unit's weight vector; plain LoRA, empirically, changes them together in a
strongly correlated way that a full fine-tune does not. Giving the magnitude its
own parameter recovers most of that missing degree of freedom at the cost of one
extra vector per layer.

Why implement this instead of depending on ``peft``:

* ``peft`` wraps the target module, which renames every base-model state-dict
  key (``q_proj.weight`` becomes ``q_proj.base_layer.weight``). At this scale a
  key rename means a bespoke conversion pass on a multi-terabyte distributed
  checkpoint.
* ``peft`` has no contract with DTensor. Its merge path calls ``.data`` on the
  base weight, which is a local shard under FSDP2 and a differently-shaped local
  shard under tensor parallelism, and it silently produces a wrong model rather
  than an error.
* An adapter is thirty lines of linear algebra. The dependency is not worth it.

Sharding — the hard part
------------------------

The base weight may be a :class:`~torch.distributed.tensor.DTensor`. Four
decisions make the adapter safe in that world:

1. :class:`LoRALinear` **subclasses** ``nn.Linear`` rather than wrapping one, so
   ``weight`` and ``bias`` stay at their original attribute names and their
   original state-dict paths. A pretrained checkpoint loads into an injected
   model with no key remapping, and every sharding plan that addresses
   ``blocks.*.attention.q_proj.weight`` keeps working unchanged.
2. The forward pass never reads ``self.weight`` directly except through
   ``F.linear``. Under FSDP2 the parameter is only unsharded inside the module's
   own forward window, so touching it anywhere else reads a shard and produces
   silently wrong numbers.
3. When the base weight is already a DTensor, the adapter factors are created as
   DTensors whose placements are *derived* from the base placements
   (see :func:`_adapter_placements`), so the low-rank product contracts along
   the same mesh dimension the base weight contracts along. Getting this wrong
   does not raise — it produces a delta that is correct on one rank and wrong on
   the others.
4. The DoRA norm is computed as a **sum of squares followed by a square root**,
   never as ``linalg.norm``. Sum is a linear reduction, so when the summed axis
   is the sharded one DTensor represents the intermediate as ``Partial`` and a
   single all-reduce finishes it exactly. ``linalg.norm`` over a sharded axis
   depends on a special-cased sharding rule that is not guaranteed to exist for
   every placement combination.

Composition order
-----------------

``apply_lora`` must run **after** tensor parallelism and **before** FSDP::

    apply_tensor_parallel(model, mesh["tp"], sequence_parallel=True)
    apply_lora(model, LoRAConfig(...))
    mark_only_lora_trainable(model)
    parallelize(model, dims, config=ParallelConfig(
        fsdp=FSDPConfig(ignore_frozen_params=True)))

After TP, because the adapter needs to see the base placements to derive its
own. Before FSDP, because ``fully_shard`` only manages parameters that exist
when it runs: an adapter registered afterwards stays an unsharded plain tensor
with no gradient reduction hook, every data-parallel rank computes a different
update, and the ranks silently diverge. :func:`apply_lora` detects that case and
refuses.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from avgen.finetune._match import matches_any, normalize_patterns

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "load_adapter",
    "lora_parameters",
    "mark_only_lora_trainable",
    "merge_lora",
    "save_adapter",
    "unmerge_lora",
]

#: Default targets: the four attention projections plus the three FFN
#: projections, named exactly as the model contract requires. Attention-only
#: adaptation is cheaper but consistently weaker on style and motion transfer,
#: which is what video fine-tuning is usually for.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Mesh dimension names that mean "this parameter is FSDP-sharded". Injecting an
#: adapter into a model sharded on one of these is the silent-divergence bug
#: described in the module docstring.
_DATA_PARALLEL_MESH_DIMS = frozenset(
    {"dp", "dp_shard", "dp_replicate", "dp_shard_cp", "dp_cp"}
)

_ADAPTER_SUFFIXES = ("lora_a", "lora_b", "lora_magnitude")


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """How a low-rank adapter is built and where it is injected.

    Args:
        rank: Inner dimension ``r``. The capacity knob. 8-16 is enough for
            style; 64-128 is what motion or subject fine-tunes need.
        alpha: Scaling numerator. The adapter contributes ``alpha/rank`` times
            the low-rank product, so raising ``rank`` at fixed ``alpha`` leaves
            the effective update magnitude — and therefore the usable learning
            rate — roughly unchanged. That decoupling is the whole reason the
            parameter exists.
        dropout: Dropout applied to the adapter input. Regularises the adapter
            without touching the frozen base path.
        target_modules: Name patterns selecting which ``nn.Linear`` modules are
            adapted, matched by :func:`avgen.finetune._match.matches_any`.
        exclude_modules: Patterns that veto a match. Applied after
            ``target_modules``, so a broad target plus a narrow exclusion is
            expressible.
        use_dora: Whether to add the per-output magnitude vector.
        init_lora_weights: ``"kaiming"``, ``"gaussian"`` or ``"zeros"``.
        rank_pattern: Per-module rank overrides as ``(pattern, rank)`` pairs,
            first match wins. A tuple of pairs rather than a mapping so the
            config stays hashable and its iteration order is reproducible.
        alpha_pattern: Per-module alpha overrides, same form.

    Raises:
        ValueError: If any numeric field is out of range or an initialisation
            scheme is unknown.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    exclude_modules: tuple[str, ...] = ()
    use_dora: bool = False
    init_lora_weights: str = "kaiming"
    rank_pattern: tuple[tuple[str, int], ...] = ()
    alpha_pattern: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate ranges and freeze the pattern tuples.

        Raises:
            ValueError: On a non-positive rank, a negative alpha, a dropout
                outside ``[0, 1)``, or an unknown initialisation scheme.
        """
        if isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError(f"rank must be a positive integer; got {self.rank!r}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be positive; got {self.alpha!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout!r}")
        if self.init_lora_weights not in {"kaiming", "gaussian", "zeros"}:
            raise ValueError(
                "init_lora_weights must be 'kaiming', 'gaussian' or 'zeros'; "
                f"got {self.init_lora_weights!r}"
            )
        object.__setattr__(
            self, "target_modules", normalize_patterns(self.target_modules)
        )
        if self.exclude_modules:
            object.__setattr__(
                self, "exclude_modules", normalize_patterns(self.exclude_modules)
            )
        for pattern, rank in self.rank_pattern:
            if isinstance(rank, bool) or rank < 1:
                raise ValueError(
                    f"rank_pattern[{pattern!r}] must be a positive integer; "
                    f"got {rank!r}"
                )
        for pattern, alpha in self.alpha_pattern:
            if alpha <= 0.0:
                raise ValueError(
                    f"alpha_pattern[{pattern!r}] must be positive; got {alpha!r}"
                )

    def rank_for(self, name: str) -> int:
        """Return the rank to use for a named module.

        Args:
            name: Dotted module name.

        Returns:
            The first matching override, or the default rank.
        """
        for pattern, rank in self.rank_pattern:
            if matches_any(name, (pattern,)):
                return rank
        return self.rank

    def alpha_for(self, name: str) -> float:
        """Return the alpha to use for a named module.

        Args:
            name: Dotted module name.

        Returns:
            The first matching override, or the default alpha.
        """
        for pattern, alpha in self.alpha_pattern:
            if matches_any(name, (pattern,)):
                return alpha
        return self.alpha

    def selects(self, name: str) -> bool:
        """Whether a module name is targeted and not excluded.

        Args:
            name: Dotted module name.

        Returns:
            Whether the module should receive an adapter.
        """
        if self.exclude_modules and matches_any(name, self.exclude_modules):
            return False
        return matches_any(name, self.target_modules)

    def to_metadata(self) -> dict[str, str]:
        """Return a JSON-safe mapping for adapter file metadata.

        Returns:
            String-to-string metadata, the only shape ``safetensors`` accepts.
        """
        return {
            "rank": str(self.rank),
            "alpha": repr(self.alpha),
            "dropout": repr(self.dropout),
            "use_dora": str(self.use_dora),
            "init_lora_weights": self.init_lora_weights,
            "target_modules": json.dumps(list(self.target_modules)),
            "exclude_modules": json.dumps(list(self.exclude_modules)),
        }


def _placements_of(tensor: torch.Tensor) -> tuple[Any, ...] | None:
    """Return a DTensor's placements, or ``None`` for a plain tensor."""
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        return tuple(tensor.placements)
    return None


def _sharded_dim(tensor: torch.Tensor) -> int | None:
    """Return the single tensor dimension a DTensor is sharded on.

    Args:
        tensor: Parameter to inspect.

    Returns:
        The sharded dimension, or ``None`` when the tensor is plain or fully
        replicated.

    Raises:
        NotImplementedError: If the tensor is sharded on more than one mesh
            dimension. Two-dimensional weight sharding (TP composed with FSDP on
            the same parameter) needs placement arithmetic this module does not
            attempt; the composition order in the module docstring avoids it.
    """
    placements = _placements_of(tensor)
    if placements is None:
        return None
    shards = [p.dim for p in placements if getattr(p, "is_shard", lambda: False)()]
    if not shards:
        return None
    if len(shards) > 1:
        raise NotImplementedError(
            "LoRA does not support a base weight sharded on two mesh dimensions "
            f"at once (placements={placements}); apply_lora must run after "
            "tensor parallelism and before FSDP"
        )
    return int(shards[0])


def _reject_fsdp_sharded(name: str, weight: torch.Tensor) -> None:
    """Refuse to adapt a weight that FSDP has already sharded.

    Args:
        name: Module name, for the error message.
        weight: The base weight.

    Raises:
        RuntimeError: If the weight lives on a data-parallel mesh dimension.
    """
    from torch.distributed.tensor import DTensor

    if not isinstance(weight, DTensor):
        return
    dim_names = set(weight.device_mesh.mesh_dim_names or ())
    if dim_names & _DATA_PARALLEL_MESH_DIMS:
        raise RuntimeError(
            f"{name}.weight is sharded on a data-parallel mesh dimension "
            f"({sorted(dim_names)}); apply_lora must run before FSDP. An adapter "
            "registered after fully_shard() is not FSDP-managed: it is never "
            "all-gathered and its gradients are never reduced, so every rank "
            "learns a different adapter and the run silently diverges"
        )


def _adapter_placements(
    weight: torch.Tensor,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """Derive DTensor placements for ``lora_a`` and ``lora_b`` from the base.

    The rule follows from where the contraction happens. For a column-wise
    parallel linear the weight is ``Shard(0)`` over output features, so ``B``
    (``outxr``) must carry the same output sharding and ``A`` (``rxin``) must be
    replicated: the low-rank product then lands ``Shard(0)`` exactly like the
    base output. For a row-wise parallel linear the weight is ``Shard(1)`` over
    input features, so ``A`` carries the input sharding and ``B`` is replicated;
    the contraction over the sharded input produces a ``Partial`` intermediate
    of width ``r``, which DTensor resolves with one small all-reduce — small
    because ``r`` is tiny, which is why the low-rank structure is cheap to
    parallelise in the first place.

    Args:
        weight: The base weight.

    Returns:
        ``(placements_for_a, placements_for_b)``, or ``None`` when the base is a
        plain tensor and the adapters should be plain too.

    Raises:
        NotImplementedError: If the base weight is sharded on a dimension other
            than 0 or 1.
    """
    from torch.distributed.tensor import Replicate, Shard

    placements = _placements_of(weight)
    if placements is None:
        return None
    dim = _sharded_dim(weight)
    if dim is None:
        replicate = (Replicate(),) * len(placements)
        return replicate, replicate
    if dim == 0:
        return (Replicate(),), (Shard(0),)
    if dim == 1:
        return (Shard(1),), (Replicate(),)
    raise NotImplementedError(
        f"base weight sharded on dim {dim}; LoRA supports dim 0 (column-wise) "
        "and dim 1 (row-wise) only"
    )


def _like(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    placements: tuple[Any, ...] | None,
) -> torch.Tensor:
    """Allocate a zero tensor matching a reference's device, dtype and layout.

    Args:
        reference: The base weight the adapter attaches to.
        shape: Global (unsharded) shape of the adapter factor.
        placements: Target placements, or ``None`` for a plain tensor.

    Returns:
        A zeroed tensor, DTensor when ``placements`` is given.
    """
    if placements is None:
        return torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    from torch.distributed.tensor import DTensor, zeros

    assert isinstance(reference, DTensor)
    return zeros(
        *shape,
        dtype=reference.dtype,
        device_mesh=reference.device_mesh,
        placements=list(placements),
    )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local view of a possibly-distributed tensor."""
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _sum_of_squares(weight: torch.Tensor) -> torch.Tensor:
    """Return the per-output-row sum of squares of a weight matrix.

    Written as an explicit sum rather than ``linalg.norm`` because summation is
    linear: when the summed axis is the sharded one, DTensor represents the
    partial sums as a ``Partial`` placement and one all-reduce completes it
    exactly. Norms are not linear, so their correctness across a shard boundary
    depends on a special-cased sharding rule existing for that exact placement
    combination.

    Args:
        weight: ``(out, in)`` weight, plain or DTensor.

    Returns:
        ``(out,)`` sums of squares, fully reduced.
    """
    from torch.distributed.tensor import DTensor, Replicate

    squares = (weight * weight).sum(dim=1)
    if isinstance(squares, DTensor) and any(p.is_partial() for p in squares.placements):
        # One all-reduce turns the per-shard partial sums into the true sum.
        # Doing it here, on an (out,) vector, is the cheapest possible place:
        # after the sqrt it would be wrong, and before the square it would be
        # an all-gather of the full weight.
        squares = squares.redistribute(
            device_mesh=squares.device_mesh,
            placements=[
                Replicate() if p.is_partial() else p for p in squares.placements
            ],
        )
    return squares


class LoRALinear(nn.Linear):
    """An ``nn.Linear`` carrying a low-rank (optionally DoRA) adapter.

    Subclassing rather than wrapping is the load-bearing design decision. A
    wrapper (``LoRALinear.base_layer.weight``) renames every base state-dict key
    the moment an adapter is attached, which means a conversion pass over a
    distributed checkpoint, a second conversion to publish the merged model, and
    a permanent fork between "adapted" and "plain" checkpoint layouts. A
    subclass keeps ``weight`` and ``bias`` exactly where they were: the same
    checkpoint loads either way, the same tensor-parallel plan addresses the
    same names, and :func:`merge_lora` restores an ordinary linear in place.

    ``lora_b`` is zero-initialised so ``B·A = 0`` and the module's output at
    initialisation is **bit-identical** to the frozen base. That is not
    cosmetic. A pretrained video model is a carefully balanced fixed point; a
    randomly initialised full-rank-``r`` perturbation of every projection at
    step 0 produces a large loss spike, and the first hundred optimizer steps
    are spent undoing damage rather than learning the target. Exact identity
    also makes the injection *testable*: if the output moves at all before the
    first optimizer step, something is wrong, and a unit test can say so.

    For DoRA the magnitude is initialised to the base row norms, which gives the
    same exact-identity property: ``m/‖W‖ = 1`` and the rescale is a no-op.

    Args:
        in_features: Input width.
        out_features: Output width.
        rank: Adapter inner dimension.
        alpha: Scaling numerator; the adapter contributes ``alpha/rank``.
        dropout: Dropout on the adapter input.
        use_dora: Whether to add the magnitude vector.
        bias: Whether the base linear has a bias.
        device: Allocation device.
        dtype: Parameter dtype.
    """

    __constants__ = ("in_features", "out_features", "rank")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_dora: bool = False,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.use_dora = bool(use_dora)
        self.scaling = self.alpha / float(self.rank)
        self.merged = False
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )
        self.lora_a = nn.Parameter(
            torch.zeros((self.rank, in_features), device=device, dtype=dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros((out_features, self.rank), device=device, dtype=dtype)
        )
        self.lora_magnitude: nn.Parameter | None = None
        if self.use_dora:
            self.lora_magnitude = nn.Parameter(
                torch.zeros((out_features,), device=device, dtype=dtype)
            )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_dora: bool = False,
        init: str = "kaiming",
    ) -> LoRALinear:
        """Build an adapted copy of an existing linear, reusing its parameters.

        The base ``weight`` and ``bias`` objects are *moved*, not copied: the
        same ``nn.Parameter`` (and therefore the same DTensor shard, the same
        storage, the same tensor-parallel placement) is re-registered on the new
        module. Copying would double peak memory at injection time on a model
        that is already sized to fill the device.

        Args:
            linear: The module to adapt.
            rank: Adapter inner dimension.
            alpha: Scaling numerator.
            dropout: Dropout on the adapter input.
            use_dora: Whether to add the magnitude vector.
            init: Initialisation scheme for ``lora_a``.

        Returns:
            An adapted module whose output currently equals ``linear``'s.
        """
        adapted = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_dora=use_dora,
            bias=linear.bias is not None,
            device="meta",
        )
        adapted.weight = linear.weight
        if linear.bias is not None:
            adapted.bias = linear.bias
        placements = _adapter_placements(linear.weight)
        a_placements = None if placements is None else placements[0]
        b_placements = None if placements is None else placements[1]
        adapted.lora_a = nn.Parameter(
            _like(linear.weight, (rank, linear.in_features), a_placements)
        )
        adapted.lora_b = nn.Parameter(
            _like(linear.weight, (linear.out_features, rank), b_placements)
        )
        adapted.reset_lora_parameters(init)
        if use_dora:
            magnitude = _like(
                linear.weight,
                (linear.out_features,),
                None if placements is None else b_placements,
            )
            adapted.lora_magnitude = nn.Parameter(magnitude)
            adapted.reset_dora_magnitude()
        return adapted

    def reset_lora_parameters(self, init: str = "kaiming") -> None:
        """Re-initialise the adapter factors, keeping ``lora_b`` at zero.

        Args:
            init: ``"kaiming"``, ``"gaussian"`` or ``"zeros"``.

        Raises:
            ValueError: On an unknown scheme.
        """
        local_a = _local(self.lora_a)
        with torch.no_grad():
            if init == "kaiming":
                # The same gain nn.Linear uses for its own weight, so the
                # adapter's input-side statistics match the layer it adapts.
                nn.init.kaiming_uniform_(local_a, a=math.sqrt(5))
            elif init == "gaussian":
                # 1/r keeps the variance of B·A independent of rank, so a rank
                # sweep does not implicitly sweep the effective learning rate.
                nn.init.normal_(local_a, std=1.0 / math.sqrt(self.rank))
            elif init == "zeros":
                # Both factors zero means both gradients are zero: the adapter
                # is permanently dead. Only useful for testing the identity
                # property in isolation.
                local_a.zero_()
            else:
                raise ValueError(
                    f"unknown init_lora_weights {init!r}; expected 'kaiming', "
                    "'gaussian' or 'zeros'"
                )
            _local(self.lora_b).zero_()

    def reset_dora_magnitude(self) -> None:
        """Set the DoRA magnitude to the base row norms, giving exact identity.

        Raises:
            RuntimeError: If the module was not built with ``use_dora``.
        """
        if self.lora_magnitude is None:
            raise RuntimeError("module has no DoRA magnitude vector")
        with torch.no_grad():
            norms = torch.sqrt(_sum_of_squares(self.weight).clamp_min(0.0))
            self.lora_magnitude.copy_(norms)

    def lora_delta(self) -> torch.Tensor:
        """Return the dense weight update ``(alpha/rank)·B·A``.

        Returns:
            ``(out, in)`` delta, matching the base weight's placements.
        """
        return (self.lora_b @ self.lora_a) * self.scaling

    def adapted_weight(self) -> torch.Tensor:
        """Return the effective weight this module applies.

        Returns:
            ``(out, in)`` weight including the adapter and, for DoRA, the
            magnitude rescale.
        """
        if self.merged:
            return self.weight
        combined = self.weight + self.lora_delta()
        if self.lora_magnitude is None:
            return combined
        return combined * self._dora_scale(combined).unsqueeze(-1)

    def _dora_scale(self, combined: torch.Tensor) -> torch.Tensor:
        """Return the per-output-row DoRA rescale ``m/‖W + ΔW‖``.

        The norm is detached. DoRA §4.3 treats the directional norm as a
        constant during backpropagation: the gradient through it is expensive
        (it needs the full weight in the backward graph, which under FSDP means
        holding the unsharded parameter alive) and empirically contributes
        almost nothing to the update. Detaching keeps the magnitude parameter's
        gradient exact while dropping only the second-order term.

        Args:
            combined: The direction matrix ``W + ΔW``.

        Returns:
            ``(out,)`` rescale factors.
        """
        assert self.lora_magnitude is not None
        # 1e-12 rather than a dtype epsilon: a zero row in a pruned or freshly
        # initialised weight must not produce inf here, and the clamp is far
        # below any magnitude that carries signal.
        norm = torch.sqrt(_sum_of_squares(combined).clamp_min(1e-12)).detach()
        return self.lora_magnitude / norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the base linear plus the adapter.

        The base path goes through ``F.linear`` on ``self.weight`` so that
        FSDP2's unshard window and any tensor-parallel placement apply exactly
        as they would to an unadapted linear. The adapter is evaluated as two
        thin matrix products rather than by materialising ``ΔW``: the dense form
        is ``outxin`` and would defeat the point of the factorisation, and under
        DoRA it is materialised only because the row norm genuinely needs it.

        Args:
            input: ``(..., in_features)`` activations.

        Returns:
            ``(..., out_features)`` activations.
        """
        if self.merged or self.lora_magnitude is not None:
            # DoRA rescales the *whole* weight, base included, so it cannot be
            # expressed as a residual added to the base output; the dense form
            # is unavoidable here.
            return functional.linear(input, self.adapted_weight(), self.bias)
        base = functional.linear(input, self.weight, self.bias)
        hidden = functional.linear(self.lora_dropout(input), self.lora_a)
        return base + functional.linear(hidden, self.lora_b) * self.scaling

    def merge(self) -> None:
        """Fold the adapter into the base weight in place.

        Raises:
            RuntimeError: If already merged.
        """
        if self.merged:
            raise RuntimeError("adapter is already merged")
        with torch.no_grad():
            merged = self.adapted_weight()
            self.weight.copy_(merged)
        self.merged = True

    def unmerge(self) -> None:
        """Undo :meth:`merge`, restoring the base weight.

        Exact for plain LoRA (subtracting the same delta). For DoRA the inverse
        divides by the rescale that was applied, which is exact in real
        arithmetic and accurate to rounding in floating point — a merged DoRA
        weight should not be round-tripped repeatedly.

        Raises:
            RuntimeError: If not currently merged.
        """
        if not self.merged:
            raise RuntimeError("adapter is not merged")
        self.merged = False
        with torch.no_grad():
            if self.lora_magnitude is None:
                self.weight.sub_(self.lora_delta())
                return
            # Recover the direction matrix, then remove the low-rank delta.
            scaled = self.weight
            norm = torch.sqrt(_sum_of_squares(scaled).clamp_min(1e-12))
            rescale = norm / self.lora_magnitude.clamp_min(1e-12)
            direction = scaled * rescale.unsqueeze(-1)
            self.weight.copy_(direction - self.lora_delta())

    def extra_repr(self) -> str:
        """Return the base repr plus the adapter shape."""
        kind = "dora" if self.lora_magnitude is not None else "lora"
        return (
            f"{super().extra_repr()}, {kind}_rank={self.rank}, "
            f"alpha={self.alpha}, merged={self.merged}"
        )


def _replace_child(root: nn.Module, name: str, module: nn.Module) -> None:
    """Install ``module`` at the dotted path ``name`` under ``root``."""
    parent_path, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child, module)


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Inject adapters into every targeted ``nn.Linear``, in place.

    Args:
        model: The model to adapt. Modified in place and also returned, so the
            call reads well either way.
        config: Which modules to adapt and how.

    Returns:
        The same model.

    Raises:
        RuntimeError: If a target weight is already FSDP-sharded (see the module
            docstring on composition order), or if no module matched — an empty
            match is almost always a typo in ``target_modules`` and silently
            training zero parameters wastes a cluster.
    """
    targets: list[tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, LoRALinear)
        and config.selects(name)
    ]
    if not targets:
        available = sorted(
            {
                name.rpartition(".")[2]
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
            }
        )
        raise RuntimeError(
            f"no nn.Linear matched target_modules={config.target_modules}; "
            f"linear submodule names in this model: {available}"
        )
    for name, linear in targets:
        _reject_fsdp_sharded(name, linear.weight)
        _replace_child(
            model,
            name,
            LoRALinear.from_linear(
                linear,
                rank=config.rank_for(name),
                alpha=config.alpha_for(name),
                dropout=config.dropout,
                use_dora=config.use_dora,
                init=config.init_lora_weights,
            ),
        )
    return model


def lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield every adapted module with its dotted name.

    Args:
        model: The model to walk.

    Yields:
        ``(name, module)`` pairs in module-registration order, which is stable.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the adapter parameters, in a stable order, for the optimizer.

    Args:
        model: The adapted model.

    Returns:
        Every ``lora_a``, ``lora_b`` and DoRA magnitude, in walk order. A list
        rather than a generator because an optimizer consumes the iterable more
        than once when it builds parameter groups.
    """
    collected: list[nn.Parameter] = []
    for _, module in lora_modules(model):
        collected.append(module.lora_a)
        collected.append(module.lora_b)
        if module.lora_magnitude is not None:
            collected.append(module.lora_magnitude)
    return collected


def mark_only_lora_trainable(
    model: nn.Module,
    *,
    train_biases: bool = False,
    extra_trainable: Sequence[str] = (),
) -> int:
    """Freeze everything except the adapters.

    Do this **before** :func:`avgen.parallel.parallelize` so that
    ``FSDPConfig(ignore_frozen_params=True)`` can see which parameters are
    frozen and skip sharding them.

    Args:
        model: The adapted model.
        train_biases: Whether base biases stay trainable. Cheap, occasionally
            worth it, and off by default because it breaks the property that the
            adapter file alone reconstructs the fine-tune.
        extra_trainable: Extra name patterns to keep trainable — usually a
            newly added head or an input projection whose shape changed.

    Returns:
        The number of trainable parameter tensors.
    """
    patterns = tuple(extra_trainable)
    trainable = 0
    for name, parameter in model.named_parameters():
        leaf = name.rpartition(".")[2]
        keep = leaf in _ADAPTER_SUFFIXES
        if not keep and train_biases and leaf == "bias":
            keep = True
        if not keep and patterns and matches_any(name, patterns):
            keep = True
        parameter.requires_grad_(keep)
        trainable += int(keep)
    return trainable


def merge_lora(model: nn.Module, *, strip: bool = False) -> nn.Module:
    """Fold every adapter into its base weight, in place.

    Merging is what makes LoRA free at inference: the deployed model is an
    ordinary linear stack with no extra matmuls, no extra memory traffic and no
    adapter-aware serving code. It is also destructive — the base weights of a
    merged model are no longer the pretrained ones — so a training loop should
    merge a *copy*, or merge, export, and :func:`unmerge_lora`.

    Args:
        model: The adapted model.
        strip: Whether to also replace each :class:`LoRALinear` with a plain
            ``nn.Linear`` at the same path, dropping the adapter parameters
            entirely. Do this for a release artefact; leave it off if you intend
            to unmerge.

    Returns:
        The same model.
    """
    for name, module in list(lora_modules(model)):
        if not module.merged:
            module.merge()
        if not strip:
            continue
        plain = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        plain.weight = module.weight
        if module.bias is not None:
            plain.bias = module.bias
        _replace_child(model, name, plain)
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Undo :func:`merge_lora` for every adapter that is currently merged.

    Args:
        model: The adapted model.

    Returns:
        The same model.
    """
    for _, module in lora_modules(model):
        if module.merged:
            module.unmerge()
    return model


def _gathered(tensor: torch.Tensor) -> torch.Tensor:
    """Return a full, replicated CPU-savable view of a possibly-sharded tensor.

    Args:
        tensor: Adapter parameter.

    Returns:
        A plain contiguous tensor holding the global value.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        # Collective: every rank must reach this, which is why save_adapter is
        # documented as a synchronous all-rank call even though only rank 0
        # writes.
        return tensor.full_tensor().detach().contiguous()
    return tensor.detach().contiguous()


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the adapter tensors, keyed by their full parameter path.

    Args:
        model: The adapted model.

    Returns:
        A mapping small enough to write as a single file — for rank 16 on a 2B
        model this is a few megabytes against tens of gigabytes for the model.

    Raises:
        RuntimeError: If any adapter is currently merged, in which case the
            factors no longer describe the difference from the saved base and
            the file would silently be a no-op adapter.
    """
    state: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model):
        if module.merged:
            raise RuntimeError(
                f"{name} is merged; call unmerge_lora(model) before saving, or "
                "the saved factors will not reproduce the adapted model"
            )
        state[f"{name}.lora_a"] = _gathered(module.lora_a)
        state[f"{name}.lora_b"] = _gathered(module.lora_b)
        if module.lora_magnitude is not None:
            state[f"{name}.lora_magnitude"] = _gathered(module.lora_magnitude)
    return state


def save_adapter(
    path: str | Path,
    model: nn.Module,
    *,
    config: LoRAConfig | None = None,
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write the adapter tensors alone to a ``safetensors`` file.

    An adapter is distributed on its own: a few megabytes that a user drops next
    to any copy of the base checkpoint. That only works if the file contains
    exactly the adapter and enough metadata to rebuild the injection, which is
    why the config travels inside the file rather than beside it.

    This is a **collective** call when the model is distributed: every rank must
    invoke it so the gathers complete, even though only rank 0 writes.

    Args:
        path: Destination file.
        model: The adapted model.
        config: The config used to build the adapter, stored as metadata.
        extra_metadata: Additional string metadata — a base-model fingerprint, a
            training step, a dataset name.

    Returns:
        The path written.

    Raises:
        RuntimeError: If the model has no adapters.
    """
    from safetensors.torch import save_file

    state = adapter_state_dict(model)
    if not state:
        raise RuntimeError("model contains no LoRALinear modules; nothing to save")
    metadata = {"format": "avgen-lora-v1"}
    if config is not None:
        metadata.update(config.to_metadata())
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_writer_rank():
        save_file(state, str(destination), metadata=metadata)
    return destination


def _is_writer_rank() -> bool:
    """Whether this process should write the adapter file."""
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def load_adapter(
    path: str | Path,
    model: nn.Module,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load adapter tensors into an already-injected model.

    The model must already carry :class:`LoRALinear` modules of the right rank —
    call :func:`apply_lora` with the config recorded in the file's metadata
    first. Loading does not build modules, so it cannot disagree with the
    parallelisation that was applied to them.

    Args:
        path: Adapter file.
        model: The model to load into.
        strict: Whether a key present in the file but absent from the model (or
            the reverse) is an error.

    Returns:
        The file's metadata mapping.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If ``strict`` and the key sets disagree.
        ValueError: If a tensor's shape does not match its destination.
    """
    from safetensors import safe_open

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"adapter file not found: {source}")
    destinations = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.rpartition(".")[2] in _ADAPTER_SUFFIXES
    }
    metadata: dict[str, Any] = {}
    loaded: set[str] = set()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
            target = destinations.get(key)
            if target is None:
                if strict:
                    raise KeyError(
                        f"adapter file contains {key!r}, which the model has no "
                        "parameter for; was apply_lora called with the same "
                        "target_modules?"
                    )
                continue
            value = handle.get_tensor(key)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"{key} has shape {tuple(value.shape)} in the file but "
                    f"{tuple(target.shape)} in the model; the ranks disagree"
                )
            with torch.no_grad():
                target.copy_(value.to(dtype=target.dtype))
            loaded.add(key)
    missing = sorted(set(destinations) - loaded)
    if strict and missing:
        raise KeyError(f"adapter file is missing parameters: {missing}")
    return metadata
