"""Pipeline parallelism: splitting model depth across ranks.

Pipeline parallelism is deliberately the **last** axis avgen reaches for, and
the documentation says so plainly because the alternative is people enabling it
first and wondering where their throughput went.

Its cost is the bubble: with ``p`` stages and ``m`` microbatches, a naive
schedule idles for roughly ``(p-1)/(m+p-1)`` of the time. You buy that back by
raising ``m``, which raises activation memory, which is the thing that was
already scarce. Interleaved and zero-bubble schedules shrink it further at the
cost of more complex stage assignment.

Its benefit is real when the model is genuinely too deep for one node's memory
even after FSDP and TP, or when the inter-node fabric is slow enough that
FSDP's all-gathers dominate — pipeline traffic is a small point-to-point
activation handoff, by far the cheapest communication pattern of the five axes.

For a video DiT, order of preference is: context parallel (sequence is the
problem), then FSDP (parameters), then tensor parallel (per-layer activations),
then pipeline. Reach for pipeline when depth alone is the constraint.

**Stage balance is the thing to get right.** Splitting a 48-block model into 4
stages of 12 blocks is only balanced if every block costs the same — which is
false when the first stage also carries the patch embedding and the last carries
the output projection. :func:`balanced_split_points` accounts for that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    _PipelineSchedule,
)

__all__ = [
    "PipelineConfig",
    "ScheduleName",
    "balanced_split_points",
    "build_pipeline_schedule",
    "split_model",
]

ScheduleName = Literal["gpipe", "1f1b", "interleaved_1f1b", "zero_bubble"]

_SCHEDULES: dict[str, type[_PipelineSchedule]] = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
    "interleaved_1f1b": ScheduleInterleaved1F1B,
    "zero_bubble": ScheduleInterleavedZeroBubble,
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """How to split a model across pipeline stages.

    Args:
        schedule: Which schedule to run. ``1f1b`` is the right default: it has
            the same bubble as GPipe but holds only ``p`` microbatches of
            activations instead of ``m``, which for a video model is the
            difference between fitting and not.
        microbatches: Microbatches per optimizer step. Must be at least the
            number of stages, and larger is better for the bubble and worse for
            memory.
        stages_per_rank: Virtual stages each rank owns. Greater than one enables
            the interleaved schedules, which cut the bubble by that factor at
            the cost of more point-to-point messages.
        embedding_weight: Relative cost of the input embedding, in units of one
            transformer block, used when balancing stages.
        head_weight: Relative cost of the output head, same units.

    Raises:
        ValueError: If the schedule is unknown or a count is below one.
    """

    schedule: ScheduleName = "1f1b"
    microbatches: int = 8
    stages_per_rank: int = 1
    embedding_weight: float = 0.5
    head_weight: float = 0.5

    def __post_init__(self) -> None:
        """Validate the schedule name and counts."""
        if self.schedule not in _SCHEDULES:
            raise ValueError(
                f"unknown schedule {self.schedule!r}; "
                f"available: {', '.join(sorted(_SCHEDULES))}"
            )
        for name in ("microbatches", "stages_per_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be >= 1; got {value!r}")
        if self.stages_per_rank > 1 and self.schedule in ("gpipe", "1f1b"):
            raise ValueError(
                f"schedule {self.schedule!r} is single-stage; use "
                "'interleaved_1f1b' or 'zero_bubble' with stages_per_rank > 1"
            )

    @property
    def is_interleaved(self) -> bool:
        """Whether this configuration runs multiple stages per rank."""
        return self.stages_per_rank > 1

    def bubble_fraction(self, stages: int) -> float:
        """Estimate the fraction of time spent in the pipeline bubble.

        A planning aid, not a measurement: it ignores communication and
        assumes perfectly balanced stages, so treat it as a lower bound on
        the real cost.

        Args:
            stages: Total pipeline stages.

        Returns:
            Estimated idle fraction in ``[0, 1)``.
        """
        if stages <= 1:
            return 0.0
        effective = stages // max(1, self.stages_per_rank)
        if self.schedule == "zero_bubble":
            # Zero-bubble schedules recover most, but not all, of the idle time.
            effective = max(1, effective // 2)
        return (effective - 1) / (self.microbatches + effective - 1)


def balanced_split_points(
    num_blocks: int,
    num_stages: int,
    *,
    embedding_weight: float = 0.5,
    head_weight: float = 0.5,
) -> list[int]:
    """Choose block indices at which to cut the model into balanced stages.

    The first stage carries the patch embedding and the last carries the output
    head, so an even split of blocks produces uneven stages. This weights those
    extras and rebalances, which matters because a pipeline runs at the speed of
    its slowest stage — a 15% overweight first stage costs 15% of the whole job.

    Args:
        num_blocks: Number of transformer blocks.
        num_stages: Number of pipeline stages.
        embedding_weight: Cost of the embedding in units of one block.
        head_weight: Cost of the head in units of one block.

    Returns:
        Sorted block indices where a new stage begins, excluding zero. The list
        has ``num_stages - 1`` entries.

    Raises:
        ValueError: If there are fewer blocks than stages.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1; got {num_stages!r}")
    if num_blocks < num_stages:
        raise ValueError(
            f"cannot split {num_blocks} blocks into {num_stages} stages; "
            "reduce the pipeline degree or use a deeper model"
        )
    if num_stages == 1:
        return []

    costs = [1.0] * num_blocks
    costs[0] += embedding_weight
    costs[-1] += head_weight
    target = sum(costs) / num_stages

    points: list[int] = []
    accumulated = 0.0
    for index, cost in enumerate(costs):
        accumulated += cost
        remaining_stages = num_stages - len(points) - 1
        remaining_blocks = num_blocks - index - 1
        if remaining_stages == 0:
            break
        # Cut when this stage has met its share, but never so late that the
        # remaining stages cannot each receive at least one block.
        if accumulated >= target or remaining_blocks == remaining_stages:
            points.append(index + 1)
            accumulated = 0.0
    return points[: num_stages - 1]


def split_model(
    model: nn.Module,
    *,
    stage_index: int,
    num_stages: int,
    split_points: Sequence[int],
    block_attribute: str = "blocks",
) -> nn.Module:
    """Return this rank's stage by deleting the blocks it does not own.

    Deleting rather than copying is deliberate: the blocks another stage owns
    were never materialised on this rank if the model was built on the meta
    device, so a large model never has to fit anywhere at once.

    Args:
        model: The full model.
        stage_index: This rank's stage.
        num_stages: Total stages.
        split_points: Block indices where stages begin, from
            :func:`balanced_split_points`.
        block_attribute: Name of the ``ModuleList`` of blocks.

    Returns:
        The model with only this stage's blocks, embedding, and head retained.

    Raises:
        AttributeError: If the block list is missing.
        ValueError: If ``stage_index`` is out of range.
    """
    if not 0 <= stage_index < num_stages:
        raise ValueError(
            f"stage_index must be in [0, {num_stages}); got {stage_index!r}"
        )
    blocks = getattr(model, block_attribute, None)
    if not isinstance(blocks, nn.ModuleList):
        raise AttributeError(
            f"model {type(model).__name__} has no nn.ModuleList {block_attribute!r}"
        )

    boundaries = [0, *split_points, len(blocks)]
    start, stop = boundaries[stage_index], boundaries[stage_index + 1]
    kept = nn.ModuleList(list(blocks)[start:stop])
    setattr(model, block_attribute, kept)

    is_first = stage_index == 0
    is_last = stage_index == num_stages - 1
    # Drop the pieces this stage does not run so their parameters are neither
    # allocated nor included in the optimizer.
    if not is_first:
        for name in ("patch_embed", "time_embed", "text_proj"):
            if hasattr(model, name):
                setattr(model, name, None)
    if not is_last:
        for name in ("final_norm", "final_proj"):
            if hasattr(model, name):
                setattr(model, name, None)
    return model


def build_pipeline_schedule(
    stages: Sequence[PipelineStage],
    config: PipelineConfig,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
) -> _PipelineSchedule:
    """Construct the pipeline schedule that drives the stages.

    Args:
        stages: This rank's stages. One entry unless interleaving.
        config: Schedule configuration.
        loss_fn: Loss applied on the last stage.

    Returns:
        The schedule, whose ``step`` runs one full optimizer step's worth of
        microbatches.

    Raises:
        ValueError: If the stage count disagrees with the configuration.
    """
    if config.is_interleaved and len(stages) != config.stages_per_rank:
        raise ValueError(
            f"interleaved schedule expects {config.stages_per_rank} stages per "
            f"rank; got {len(stages)}"
        )
    schedule_class = _SCHEDULES[config.schedule]
    if config.is_interleaved:
        return schedule_class(
            list(stages),
            n_microbatches=config.microbatches,
            loss_fn=loss_fn,
        )
    return schedule_class(
        stages[0],
        n_microbatches=config.microbatches,
        loss_fn=loss_fn,
    )


def stage_for_rank(mesh: DeviceMesh) -> tuple[int, int]:
    """Return ``(stage_index, num_stages)`` for this rank.

    Args:
        mesh: The ``pp`` sub-mesh.

    Returns:
        This rank's stage index and the total number of stages.
    """
    return mesh.get_local_rank(), mesh.size()
