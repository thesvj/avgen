"""Orchestration: generate, measure, gather across ranks, and pin the result.

Three things this module gets right that an ad-hoc evaluation script usually
does not.

**Sharding is on the data rank only.** Prompts are split across ``data_rank``,
exactly as training data is, because context- and tensor-parallel ranks hold
shards of the *same* sample and must be handed the *same* prompt. Sharding on
global rank instead produces a run where each prompt is evaluated
``cp * tp`` times and the sample count in the report is wrong by that factor —
silently, and in the direction that makes Fréchet-style metrics look better.

**Reduction is over sufficient statistics, not over per-rank means.** Each rank
contributes ``(sum, count)`` pairs that are summed globally. Averaging per-rank
``compute()`` outputs is wrong whenever ranks saw different sample counts, which
they do the moment the prompt count is not divisible by the data world size —
i.e. almost always, because prompt sets are round numbers and world sizes are
powers of two.

**A metric that could not run is recorded, not dropped.** A pixel-only metric
evaluated on latents, a metric whose input the run did not produce, a gated
metric whose backend is missing: each lands in the report's ``skipped`` section
with its reason. An absent row reads as "not measured"; this makes it say
"could not be measured, here is why".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from avgen.eval.learned import MissingBackendError, list_learned_metrics
from avgen.eval.protocols import MetricError, RunningMetric, build_metric, list_metrics
from avgen.eval.report import EvalPin, EvalReport, prompt_set_digest

__all__ = [
    "EvalBatch",
    "MetricBundle",
    "build_metrics",
    "run_eval_suite",
    "shard_prompts",
]

#: Deterministic fallback prompt set. Fine for a smoke test, and meaningless as
#: a reported number — the prompts are trivially short and cover no interesting
#: compositional structure. Any real evaluation supplies its own file.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "a red balloon rising over a field",
    "waves breaking on a rocky shore",
    "a cat walking across a wooden floor",
    "steam rising from a cup of coffee",
    "a bicycle wheel spinning slowly",
    "leaves moving in the wind",
    "a candle flame flickering in a dark room",
    "traffic crossing a bridge at dusk",
)


@dataclass(slots=True)
class EvalBatch:
    """One batch of generated media and everything a metric might want with it.

    Args:
        video: ``(batch, channels, frames, height, width)`` generated content.
        prompts: The prompt for each element.
        audio: ``(batch, channels, frames)`` generated audio, when present.
        reference: Ground-truth or conditioning content, for the conditional
            metrics.
        mask: ``(batch, 1, frames, height, width)`` generated-region mask, for
            inpainting.
        extra: Anything else metrics accept, passed straight through.
    """

    video: torch.Tensor
    prompts: tuple[str, ...] = ()
    audio: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Return the keyword bundle handed to every metric's ``update``.

        Returns:
            A mapping with ``None`` entries removed, so a metric's
            required-input check reports a genuinely absent input rather than
            a present-but-empty one.
        """
        inputs: dict[str, Any] = {"video": self.video, **self.extra}
        if self.prompts:
            inputs["prompts"] = self.prompts
        for name, value in (
            ("audio", self.audio),
            ("reference", self.reference),
            ("mask", self.mask),
        ):
            if value is not None:
                inputs[name] = value
        return inputs


@dataclass(slots=True)
class MetricBundle:
    """The metrics that will run, and the ones that will not, with reasons.

    Args:
        metrics: Name to live metric instance.
        skipped: Name to the reason it is not running.
        parameters: Name to a rendering of its configuration, recorded in the
            report's pin so that a metric reconfigured between two runs makes
            them incomparable — which it does.
    """

    metrics: dict[str, RunningMetric] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)


def build_metrics(
    names: Sequence[str],
    *,
    decoded: bool,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> MetricBundle:
    """Instantiate the requested metrics, recording why any could not be built.

    Args:
        names: Metric names from ``eval.metrics``.
        decoded: Whether evaluation runs on decoded pixels. Pixel-only metrics
            are skipped with a reason when this is false, rather than reporting
            a number whose name does not describe what it measured.
        options: Per-metric constructor arguments.

    Returns:
        The bundle.

    Raises:
        MetricError: If a name is neither a built-in nor a declared learned
            metric. An unknown metric name is a typo in a config, and running
            the evaluation without it is exactly the silent-omission failure
            this subsystem exists to prevent.
    """
    settings = dict(options or {})
    bundle = MetricBundle()
    for name in names:
        kwargs = dict(settings.get(name, {}))
        if name in list_learned_metrics():
            # Learned metrics are gated: build them through their own path so
            # a missing backend produces the install command, never a
            # substitution.
            from avgen.eval.learned import build_learned_metric

            try:
                bundle.metrics[name] = build_learned_metric(name, **kwargs)
            except MissingBackendError as error:
                bundle.skipped[name] = str(error).splitlines()[0]
                continue
        else:
            if name not in list_metrics():
                raise MetricError(
                    f"unknown metric {name!r}; built-ins: "
                    f"{', '.join(list_metrics())}; gated: "
                    f"{', '.join(list_learned_metrics())}"
                )
            metric = build_metric(name, **kwargs)
            if metric.pixel_space_only and not decoded:
                bundle.skipped[name] = (
                    "pixel-space metric requested while evaluating latents; set "
                    "eval.decode=true or drop the metric — its value on latents "
                    "would not measure what its name says"
                )
                continue
            bundle.metrics[name] = metric
        bundle.parameters[name] = (
            ",".join(f"{key}={value!r}" for key, value in sorted(kwargs.items()))
            or "default"
        )
    return bundle


def shard_prompts(
    prompts: Sequence[str],
    *,
    data_rank: int,
    data_world: int,
) -> tuple[str, ...]:
    """Split a prompt set across data-parallel ranks.

    Strided rather than contiguous. A contiguous split hands the first rank the
    first block of prompts, and prompt files are almost always grouped by
    category — so rank 0 gets every landscape and rank 3 gets every portrait,
    and any per-rank diagnostic becomes a comparison of categories rather than
    of ranks. Striding removes that confound for free.

    Args:
        prompts: The full set, in a stable order.
        data_rank: This rank's index within the data-parallel product.
        data_world: Size of the data-parallel product.

    Returns:
        This rank's prompts.

    Raises:
        ValueError: If the coordinates are inconsistent.
    """
    if data_world < 1:
        raise ValueError(f"data_world must be >= 1; got {data_world!r}")
    if not 0 <= data_rank < data_world:
        raise ValueError(
            f"data_rank must be in [0, {data_world}); got {data_rank!r}"
        )
    return tuple(prompts[data_rank::data_world])


def _gather_states(
    bundle: MetricBundle,
    *,
    gather: Callable[[Any], list[Any]] | None,
) -> None:
    """Merge every rank's sufficient statistics into this rank's metrics."""
    if gather is None:
        return
    local = {name: metric.state() for name, metric in bundle.metrics.items()}
    for remote in gather(local):
        if remote is local or not isinstance(remote, dict):
            continue
        for name, state in remote.items():
            metric = bundle.metrics.get(name)
            if metric is not None:
                metric.merge(state)


def run_eval_suite(
    batches: Iterable[EvalBatch],
    *,
    metrics: Sequence[str] | MetricBundle,
    pin: EvalPin,
    prompts: Sequence[str] = (),
    decoded: bool = False,
    metric_options: Mapping[str, Mapping[str, Any]] | None = None,
    gather: Callable[[Any], list[Any]] | None = None,
    run_name: str = "",
    step: int = 0,
    notes: str = "",
) -> EvalReport:
    """Run metrics over generated batches and produce a pinned report.

    The generation itself is the caller's job — ``avgen eval`` drives
    :class:`avgen.infer.GenerationPipeline` and hands the results here — so this
    function is equally usable on media produced elsewhere, which is what makes
    it possible to evaluate a baseline model under avgen's pin.

    Args:
        batches: Generated media, one batch at a time. Consumed lazily, so an
            evaluation set larger than memory is fine.
        metrics: Metric names, or a pre-built :class:`MetricBundle`.
        pin: The sampling settings these numbers came from. ``num_samples`` and
            ``prompt_digest`` are filled in from what was actually seen, so a
            caller cannot record a sample count that does not match the data.
        prompts: The full prompt set, for the report's digest. Pass the
            *global* set even on a sharded run; the digest identifies the
            evaluation, not this rank's slice.
        decoded: Whether the media are decoded pixels.
        metric_options: Per-metric constructor arguments.
        gather: Callable performing an all-gather of a picklable object across
            the data-parallel group, e.g.
            ``functools.partial(avgen.parallel.gather_object)``. ``None`` runs
            single-process.
        run_name: Training run being evaluated.
        step: Training step of the evaluated checkpoint.
        notes: Free text carried into the report.

    Returns:
        The report, with a pin describing exactly what produced it.

    Raises:
        MetricError: If a metric name is unknown, or a metric's required input
            was never supplied by any batch.
    """
    bundle = (
        metrics
        if isinstance(metrics, MetricBundle)
        else build_metrics(metrics, decoded=decoded, options=metric_options)
    )

    seen = 0
    fed: set[str] = set()
    for batch in batches:
        inputs = batch.as_inputs()
        seen += int(batch.video.shape[0])
        for name, metric in bundle.metrics.items():
            missing = [key for key in metric.required_inputs if key not in inputs]
            if missing:
                # Recorded once, not per batch: a metric that wants audio in a
                # video-only run should say so exactly once.
                bundle.skipped.setdefault(
                    name,
                    f"needs {', '.join(missing)}, which this evaluation did not "
                    "produce",
                )
                continue
            metric.update(**inputs)
            fed.add(name)

    for name in list(bundle.metrics):
        if name not in fed:
            bundle.metrics.pop(name)
            bundle.skipped.setdefault(name, "no batch supplied its required inputs")

    _gather_states(bundle, gather=gather)

    values: dict[str, float] = {}
    for name, metric in bundle.metrics.items():
        computed = metric.compute()
        if not computed:
            bundle.skipped.setdefault(
                name,
                "produced no statistics — every batch was too short for it "
                "(most of these metrics need at least 2-3 frames)",
            )
            continue
        values.update(computed)

    total = seen
    if gather is not None:
        total = sum(int(count) for count in gather(seen))

    resolved_pin = EvalPin(
        checkpoint=pin.checkpoint,
        sampler=pin.sampler,
        steps=pin.steps,
        guidance=pin.guidance,
        guidance_rescale=pin.guidance_rescale,
        negative_prompt=pin.negative_prompt,
        seed=pin.seed,
        num_samples=total,
        prompt_digest=prompt_set_digest(prompts) if prompts else pin.prompt_digest,
        frames=pin.frames,
        height=pin.height,
        width=pin.width,
        decoded=decoded,
        codec=pin.codec,
        precision=pin.precision,
        metrics=dict(bundle.parameters),
    )
    return EvalReport(
        pin=resolved_pin,
        metrics=values,
        skipped=dict(bundle.skipped),
        run_name=run_name,
        step=step,
        notes=notes,
    )
