"""Selecting which parameters train, and accounting for what that costs.

Freezing is the cheapest fine-tuning tool there is and the easiest to get
subtly wrong. Two failure modes account for most of the wasted compute:

* **Freezing the wrong thing and not noticing.** A typo in a pattern that
  matches nothing leaves the whole model trainable — the job runs, the loss
  falls, and the "adapter fine-tune" quietly cost a full training run's memory.
  Every function here reports what it matched, and
  :func:`trainable_summary` exists so a run can log the ratio at step zero and
  a human can see immediately whether it is the number they intended.

* **Freezing without telling the sharding layer.** ``requires_grad=False``
  removes a parameter from the optimizer, not from FSDP. A sharded frozen base
  is still all-gathered before every block's forward and freed after it, every
  step, forever, for weights that will never change. On a LoRA fine-tune that is
  99% of the model's communication volume spent reconstructing constants.
  :class:`~avgen.parallel.fsdp.FSDPConfig` has ``ignore_frozen_params`` for
  exactly this: it excludes frozen parameters from sharding entirely, so each
  rank simply keeps its own resident copy and the all-gather disappears.

  The trade is memory for bandwidth: an ignored parameter is replicated on every
  rank rather than sharded across them. That is the right trade whenever the
  frozen part fits — which, for a LoRA fine-tune of a model that was pretrained
  on the same hardware, it does by construction. It is the wrong trade when the
  frozen base alone exceeds device memory, and then ``ignore_frozen_params``
  must stay off and the all-gather is the price of running at all.

  Ordering: freeze first, ``parallelize`` second. ``apply_fsdp`` reads
  ``requires_grad`` at wrap time, so a parameter frozen afterwards is already
  sharded and the flag has no effect.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from avgen.finetune._match import matches_any, normalize_patterns

__all__ = [
    "TrainableSummary",
    "freeze_all",
    "freeze_except",
    "freeze_matching",
    "trainable_summary",
]


@dataclass(frozen=True, slots=True)
class TrainableSummary:
    """Parameter accounting for a partially frozen model.

    Counts are in *global* elements: for a DTensor, ``numel()`` reports the full
    logical size rather than the rank-local shard, so the summary reads the same
    on one GPU and on a thousand. That is what makes it comparable across runs.

    Args:
        trainable: Elements with ``requires_grad=True``.
        frozen: Elements with ``requires_grad=False``.
        trainable_tensors: Number of trainable parameter tensors.
        total_tensors: Number of parameter tensors.
    """

    trainable: int
    frozen: int
    trainable_tensors: int
    total_tensors: int

    @property
    def total(self) -> int:
        """Total parameter elements."""
        return self.trainable + self.frozen

    @property
    def percentage(self) -> float:
        """Percentage of elements that train, or ``0.0`` for an empty model."""
        return 100.0 * self.trainable / self.total if self.total else 0.0

    @property
    def optimizer_state_bytes(self) -> int:
        """Bytes of Adam moment state implied by the trainable count.

        Two fp32 moments per trainable element. This is usually the number that
        decides whether a fine-tune fits, and it is the number people forget:
        the parameters themselves may be bf16, but the moments are not.

        Returns:
            Estimated optimizer state size in bytes.
        """
        return self.trainable * 8

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns:
            A log line of the form
            ``"trainable 4.7M / 2.10B (0.22%), 288/1105 tensors"``.
        """
        return (
            f"trainable {_si(self.trainable)} / {_si(self.total)} "
            f"({self.percentage:.4g}%), "
            f"{self.trainable_tensors}/{self.total_tensors} tensors"
        )


def _si(value: int) -> str:
    """Format an element count with a metric suffix."""
    for limit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= limit:
            return f"{value / limit:.3g}{suffix}"
    return str(value)


def freeze_all(model: nn.Module) -> int:
    """Freeze every parameter.

    Args:
        model: The model to freeze.

    Returns:
        The number of parameter tensors frozen.
    """
    count = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        count += 1
    return count


def freeze_except(model: nn.Module, patterns: Iterable[str]) -> TrainableSummary:
    """Freeze everything whose name does not match one of ``patterns``.

    The whitelist form. Use it when you know exactly what should move — the
    final block, a newly added head, the adapters — which is the common case in
    fine-tuning and the safe default, because anything you forget to name stays
    frozen rather than silently training.

    Args:
        model: The model to modify in place.
        patterns: Parameter-name patterns to keep trainable, matched by
            :func:`avgen.finetune._match.matches_any`. Patterns address
            *parameter* names (``blocks.7.attention.q_proj.weight``), so a
            module-level pattern such as ``"blocks.7"`` selects that block's
            whole parameter set via the component rule.

    Returns:
        The resulting accounting.

    Raises:
        TypeError: If a bare string is passed instead of a sequence.
        ValueError: If the pattern collection is empty, or if no parameter
            matched. An empty match means the whole model was frozen and the
            optimizer would have nothing to step, which is never intended.
    """
    keep = normalize_patterns(patterns)
    matched = 0
    for name, parameter in model.named_parameters():
        trainable = matches_any(name, keep)
        parameter.requires_grad_(trainable)
        matched += int(trainable)
    if matched == 0:
        raise ValueError(
            f"freeze_except({keep}) matched no parameter, so the entire model "
            "is frozen; check the patterns against model.named_parameters()"
        )
    return trainable_summary(model)


def freeze_matching(model: nn.Module, patterns: Iterable[str]) -> TrainableSummary:
    """Freeze the parameters that match ``patterns``, leaving the rest alone.

    The blacklist form, and deliberately non-destructive: parameters that do not
    match keep whatever ``requires_grad`` they already had. That composes — a
    LoRA injection followed by ``freeze_matching(model, ("text_proj",))`` freezes
    an extra tower without re-enabling the base the adapter just froze.

    Args:
        model: The model to modify in place.
        patterns: Parameter-name patterns to freeze.

    Returns:
        The resulting accounting.

    Raises:
        TypeError: If a bare string is passed instead of a sequence.
        ValueError: If the pattern collection is empty, or if no parameter
            matched — a no-op freeze is a typo, not a valid request.
    """
    drop = normalize_patterns(patterns)
    matched = 0
    for name, parameter in model.named_parameters():
        if matches_any(name, drop):
            parameter.requires_grad_(False)
            matched += 1
    if matched == 0:
        raise ValueError(
            f"freeze_matching({drop}) matched no parameter; check the patterns "
            "against model.named_parameters()"
        )
    return trainable_summary(model)


def trainable_summary(model: nn.Module) -> TrainableSummary:
    """Count trainable against total parameters.

    Args:
        model: The model to inspect.

    Returns:
        The accounting, including the percentage and the implied Adam state.
    """
    trainable = 0
    frozen = 0
    trainable_tensors = 0
    total_tensors = 0
    for parameter in model.parameters():
        elements = parameter.numel()
        total_tensors += 1
        if parameter.requires_grad:
            trainable += elements
            trainable_tensors += 1
        else:
            frozen += elements
    return TrainableSummary(
        trainable=trainable,
        frozen=frozen,
        trainable_tensors=trainable_tensors,
        total_tensors=total_tensors,
    )
