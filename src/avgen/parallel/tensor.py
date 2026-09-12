"""Tensor parallelism, including sequence parallelism on the norm regions.

Tensor parallelism splits individual weight matrices across ranks. For a
transformer block the classic decomposition is:

* Attention ``q/k/v`` projections: **column-wise**. Each rank owns a slice of
  the heads, computes attention for those heads independently, and no
  communication is needed *inside* attention at all.
* Attention output projection: **row-wise**. Each rank multiplies its head slice
  and the results are all-reduced into the full output.
* Feed-forward ``gate``/``up``: column-wise; ``down``: row-wise. Same pattern.

That gives two all-reduces per block — one after attention, one after the
feed-forward — and shards the largest weights by ``tp``.

**Sequence parallelism is the other half, and it is not optional.** Plain TP
leaves the norm and residual regions replicated, so every rank holds the full
``(batch, seq, width)`` activation there. For a language model that is a
rounding error. For video, where ``seq`` is 100k, it is the dominant term and it
cancels out most of TP's memory benefit. Sharding those regions along the
sequence turns the two all-reduces into two reduce-scatter/all-gather pairs at
identical bandwidth cost, and shards the activation as well as the weights.

**Models declare their own plan.** A model implements
:class:`TensorParallelizable` and returns a mapping from submodule name to
:class:`~torch.distributed.tensor.parallel.ParallelStyle`. The framework never
guesses from module names, because guessing wrong produces a model that trains
to a plausible-looking but incorrect loss.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

__all__ = [
    "TensorParallelPlan",
    "TensorParallelizable",
    "apply_tensor_parallel",
    "standard_block_plan",
    "standard_root_plan",
]

#: A mapping from submodule path to the style that parallelises it.
TensorParallelPlan = dict[str, ParallelStyle]


@runtime_checkable
class TensorParallelizable(Protocol):
    """A model that knows how to shard itself across a tensor-parallel mesh."""

    def tensor_parallel_plan(
        self,
        *,
        sequence_parallel: bool,
    ) -> tuple[TensorParallelPlan, TensorParallelPlan]:
        """Return the ``(root_plan, block_plan)`` for this architecture.

        Args:
            sequence_parallel: Whether norm and residual regions should be
                sharded along the sequence dimension.

        Returns:
            The plan for the root module and the plan applied to every
            transformer block.
        """
        ...


def standard_block_plan(
    *,
    sequence_parallel: bool = True,
    attention_prefix: str = "attention",
    feedforward_prefix: str = "feed_forward",
    cross_attention_prefix: str | None = "cross_attention",
) -> TensorParallelPlan:
    """Build the conventional plan for a pre-norm transformer block.

    Covers any block laid out as norm → attention → norm → feed-forward, with
    optional text cross-attention. Models with an unusual block should write
    their own plan rather than bending their module names to fit this one.

    Args:
        sequence_parallel: Whether to shard norms along the sequence.
        attention_prefix: Submodule name of self-attention.
        feedforward_prefix: Submodule name of the feed-forward network.
        cross_attention_prefix: Submodule name of text cross-attention, or
            ``None`` if the block has none.

    Returns:
        The block plan.
    """
    # Under sequence parallelism the block's input arrives sharded on the
    # sequence dimension, so the module that feeds attention must all-gather it
    # back to a replicated layout before the column-wise projections.
    activation_in = Shard(1) if sequence_parallel else Replicate()

    plan: TensorParallelPlan = {}

    if sequence_parallel:
        plan["attention_norm"] = SequenceParallel()
        plan["ffn_norm"] = SequenceParallel()

    plan[attention_prefix] = PrepareModuleInput(
        input_layouts=(activation_in, None),
        desired_input_layouts=(Replicate(), None),
    )
    plan[f"{attention_prefix}.q_proj"] = ColwiseParallel()
    plan[f"{attention_prefix}.k_proj"] = ColwiseParallel()
    plan[f"{attention_prefix}.v_proj"] = ColwiseParallel()
    plan[f"{attention_prefix}.out_proj"] = RowwiseParallel(output_layouts=activation_in)

    plan[feedforward_prefix] = PrepareModuleInput(
        input_layouts=(activation_in,),
        desired_input_layouts=(Replicate(),),
    )
    plan[f"{feedforward_prefix}.gate_proj"] = ColwiseParallel()
    plan[f"{feedforward_prefix}.up_proj"] = ColwiseParallel()
    plan[f"{feedforward_prefix}.down_proj"] = RowwiseParallel(
        output_layouts=activation_in
    )

    if cross_attention_prefix is not None:
        if sequence_parallel:
            plan["cross_norm"] = SequenceParallel()
        plan[cross_attention_prefix] = PrepareModuleInput(
            input_layouts=(activation_in, None, None),
            desired_input_layouts=(Replicate(), Replicate(), None),
        )
        plan[f"{cross_attention_prefix}.q_proj"] = ColwiseParallel()
        plan[f"{cross_attention_prefix}.k_proj"] = ColwiseParallel()
        plan[f"{cross_attention_prefix}.v_proj"] = ColwiseParallel()
        plan[f"{cross_attention_prefix}.out_proj"] = RowwiseParallel(
            output_layouts=activation_in
        )

    return plan


def standard_root_plan(*, sequence_parallel: bool = True) -> TensorParallelPlan:
    """Build the conventional plan for the module surrounding the blocks.

    The input projection produces the sequence that blocks consume, and the
    output projection consumes it; under sequence parallelism both must agree
    with the blocks about where the shard boundary is.

    Args:
        sequence_parallel: Whether the block interior is sequence-sharded.

    Returns:
        The root plan.
    """
    if not sequence_parallel:
        return {}
    return {
        "patch_embed": ColwiseParallel(output_layouts=Shard(1)),
        "final_norm": SequenceParallel(),
        "final_proj": ColwiseParallel(
            input_layouts=Shard(1), output_layouts=Replicate()
        ),
    }


def apply_tensor_parallel(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    sequence_parallel: bool = True,
    block_attribute: str = "blocks",
    root_plan: TensorParallelPlan | None = None,
    block_plan: TensorParallelPlan | None = None,
) -> nn.Module:
    """Parallelise a model across a tensor-parallel mesh.

    If the model implements :class:`TensorParallelizable`, its own plan is used.
    Otherwise the caller must supply plans explicitly; the framework will not
    infer a plan from module names.

    Args:
        model: The model to parallelise, modified in place.
        mesh: The ``tp`` sub-mesh.
        sequence_parallel: Whether to shard norm regions along the sequence.
        block_attribute: Name of the ``ModuleList`` of transformer blocks.
        root_plan: Explicit root plan, overriding the model's own.
        block_plan: Explicit block plan, overriding the model's own.

    Returns:
        The same model, parallelised in place.

    Raises:
        TypeError: If no plan is available from either the model or the caller.
        AttributeError: If the block list is missing.
    """
    if root_plan is None or block_plan is None:
        if not isinstance(model, TensorParallelizable):
            raise TypeError(
                f"{type(model).__name__} does not implement tensor_parallel_plan(); "
                "either implement it or pass root_plan= and block_plan= explicitly. "
                "avgen will not guess a sharding plan from module names, because a "
                "wrong guess trains silently to a wrong result"
            )
        derived_root, derived_block = model.tensor_parallel_plan(
            sequence_parallel=sequence_parallel
        )
        root_plan = root_plan if root_plan is not None else derived_root
        block_plan = block_plan if block_plan is not None else derived_block

    if root_plan:
        parallelize_module(model, mesh, root_plan)

    blocks = getattr(model, block_attribute, None)
    if not isinstance(blocks, nn.ModuleList):
        raise AttributeError(
            f"model {type(model).__name__} has no nn.ModuleList attribute "
            f"{block_attribute!r}"
        )
    for block in blocks:
        parallelize_module(block, mesh, dict(block_plan))
    return model
