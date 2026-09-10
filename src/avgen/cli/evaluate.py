"""``avgen eval`` — measure a checkpoint, and refuse invalid comparisons.

Two modes, both of which end in a pinned :class:`~avgen.eval.EvalReport`:

**Measure.** Generate from a checkpoint under the config's pinned sampling
settings, run the requested metrics, and write a report. Distributed-aware:
prompts shard on ``data_rank`` and per-metric sufficient statistics are gathered
and summed, so the numbers match what a single process would have produced.

**Compare.** Given ``--baseline``, diff two reports — and refuse when their pins
differ. That refusal is the point of the command. Most reported
video-generation comparisons are invalid because the two numbers came from
different steps, guidance, samplers, or prompt sets, and none of it was stated.
Here it raises, and ``--allow`` names the one variable you are deliberately
sweeping.

``--list-metrics`` prints every metric with what it measures and whether its
backend is present, because choosing metrics from a list of names alone is how
people end up reporting temporal consistency without motion magnitude and
concluding that a frozen video is a good one.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from avgen.cli._common import (
    add_traceback_argument,
    emit,
    fail,
    load_run_config,
    rule,
)

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``eval`` subcommand.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "eval",
        help="evaluate a checkpoint and write a pinned report",
        description=(
            "Generate under pinned sampling settings, compute metrics, and "
            "write a report that records exactly what produced the numbers.\n\n"
            "With --baseline, compare against a previous report. The comparison "
            "REFUSES when the two pins differ — different steps, guidance, "
            "sampler, seed, prompt set, or resolution. Pass --allow FIELD for a "
            "deliberate one-variable sweep."
        ),
        epilog=(
            "examples:\n"
            "  avgen eval --checkpoint runs/base/ckpt-40000 "
            "--config configs/eval/default.yaml\n"
            "  avgen eval --checkpoint ckpt-40000 --config configs/eval/default.yaml "
            "--baseline runs/base/eval-20000.json\n"
            "  avgen eval --baseline a.json --report b.json --allow seed\n"
            "  avgen eval --list-metrics\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default="",
        help="checkpoint to evaluate",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="",
        help="run configuration supplying eval.* settings",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="dotted overrides, e.g. eval.steps=50 eval.guidance=7.5",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="",
        help="where to write the report; empty uses eval.output_dir",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="training step of the evaluated checkpoint, recorded in the report",
    )

    comparison = parser.add_argument_group("comparison")
    comparison.add_argument(
        "--baseline",
        metavar="PATH",
        default="",
        help="a previous report to compare against",
    )
    comparison.add_argument(
        "--report",
        metavar="PATH",
        default="",
        help=(
            "compare this existing report against --baseline instead of "
            "generating. Use it to diff two runs after the fact."
        ),
    )
    comparison.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="FIELD",
        help=(
            "pinned field permitted to differ, e.g. --allow seed for a seed "
            "sweep. Repeatable. Use it for a deliberate one-variable study and "
            "nothing else — every use widens what the comparison can mean."
        ),
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="list every metric, what it measures, and whether it can run",
    )
    add_traceback_argument(parser)
    parser.set_defaults(handler=run)
    return parser


def _list_metrics() -> int:
    """Print the metric catalogue."""
    from avgen.eval import describe_learned_metrics, describe_metrics

    emit(rule("DEPENDENCY-FREE METRICS", character="="))
    emit("  Always available. Read them in pairs — every one is trivially")
    emit("  maximisable on its own (a frozen frame wins temporal consistency).")
    emit()
    described = describe_metrics()
    width = max(len(name) for name in described) if described else 0
    for name, description in described.items():
        emit(f"  {name.ljust(width)}  {description}")
    emit()
    emit(rule("GATED METRICS", character="="))
    emit("  Need a pretrained backend. avgen refuses these when the backend is")
    emit("  missing; it never substitutes a different metric under the name.")
    emit()
    for name, detail in describe_learned_metrics().items():
        emit(f"  {name}")
        emit(f"      measures: {detail['measures']}")
        emit(f"      caveat:   {detail['caveat']}")
        emit(f"      status:   {detail['status']}")
        emit(f"      install:  {detail['requires']}")
    return 0


def _compare(arguments: argparse.Namespace) -> int:
    """Compare two existing reports, refusing on a pin mismatch."""
    from avgen.eval import EvalReport, IncomparableReports, compare_reports
    from avgen.eval.report import format_comparison

    baseline = EvalReport.load(arguments.baseline)
    candidate = EvalReport.load(arguments.report)
    try:
        comparison = compare_reports(baseline, candidate, allow=tuple(arguments.allow))
    except IncomparableReports as error:
        return fail(str(error))

    emit(rule("COMPARISON", character="="))
    emit(f"  settings   {baseline.pin.describe()}")
    emit(
        f"  baseline   {baseline.run_name or arguments.baseline} "
        f"(step {baseline.step}) · {baseline.pin.checkpoint}"
    )
    emit(
        f"  candidate  {candidate.run_name or arguments.report} "
        f"(step {candidate.step}) · {candidate.pin.checkpoint}"
    )
    if arguments.allow:
        emit(
            f"  ALLOWED TO DIFFER: {', '.join(arguments.allow)} — the comparison "
            "is conditional on that."
        )
    emit()
    emit(format_comparison(comparison))
    return 0


def run(arguments: argparse.Namespace) -> int:
    """Evaluate a checkpoint, or compare two existing reports.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    if arguments.list_metrics:
        return _list_metrics()

    if arguments.report:
        if not arguments.baseline:
            return fail("--report needs --baseline to compare against")
        return _compare(arguments)

    if not arguments.checkpoint:
        return fail(
            "--checkpoint is required to evaluate. To compare two existing "
            "reports instead, pass --baseline and --report."
        )

    configuration = load_run_config(arguments)
    settings = configuration.eval
    output = Path(arguments.output or settings.output_dir or configuration.output_dir)

    emit(rule("avgen eval", character="="))
    emit(f"  checkpoint  {arguments.checkpoint}")
    emit(
        f"  sampling    {settings.sampler} x{settings.steps} "
        f"cfg={settings.guidance} seed={settings.seed} "
        f"{settings.frames}x{settings.height}x{settings.width} "
        f"{'pixels' if settings.decode else 'latents'}"
    )
    emit(f"  metrics     {', '.join(settings.metrics)}")
    emit()

    report = _generate_and_measure(configuration, arguments)
    emit(report.render())

    destination = output / f"eval-step-{arguments.step}.json"
    report.save(destination)
    emit()
    emit(f"  report written to {destination}")

    if arguments.baseline:
        from avgen.eval import EvalReport, IncomparableReports, compare_reports
        from avgen.eval.report import format_comparison

        baseline = EvalReport.load(arguments.baseline)
        try:
            comparison = compare_reports(baseline, report, allow=tuple(arguments.allow))
        except IncomparableReports as error:
            return fail(str(error))
        emit()
        emit(rule("VERSUS BASELINE"))
        emit(format_comparison(comparison))
    return 0


def _generate_and_measure(configuration: Any, arguments: argparse.Namespace) -> Any:
    """Generate under the pinned settings and run the metric suite.

    Sharding and reduction follow the same rules as training: prompts split on
    ``data_rank`` only, and metric statistics are gathered as ``(sum, count)``
    pairs rather than as per-rank means — averaging per-rank means is wrong the
    moment the prompt count is not divisible by the data world size, which is
    almost always.
    """
    from avgen.cli._wiring import build_generation_pipeline
    from avgen.eval import EvalBatch, EvalPin, run_eval_suite, shard_prompts

    settings = configuration.eval
    prompts = _load_prompts(settings)

    gather: Callable[[Any], list[Any]] | None = None
    data_rank, data_world = 0, 1
    try:
        from avgen.parallel.env import init_distributed, is_distributed_launch

        if is_distributed_launch():
            from avgen.config.resolve import build_parallel_dims
            from avgen.parallel.comm import all_gather_object

            env = init_distributed()
            dims = build_parallel_dims(configuration, world_size=env.world_size)
            mesh = dims.build_mesh(env.device.type)
            data_rank, data_world = dims.data_coordinates(mesh)

            gather = all_gather_object

    except Exception:
        data_rank, data_world, gather = 0, 1, None

    local_prompts = shard_prompts(prompts, data_rank=data_rank, data_world=data_world)

    pipeline = build_generation_pipeline(configuration, arguments.checkpoint)

    def batches() -> Any:
        size = max(1, settings.batch_size)
        for start in range(0, len(local_prompts), size):
            chunk = tuple(local_prompts[start : start + size])
            media = pipeline(
                chunk,
                steps=settings.steps,
                guidance=settings.guidance,
                seed=settings.seed,
                negative_prompt=settings.negative_prompt,
                frames=settings.frames,
                height=settings.height,
                width=settings.width,
            )
            yield EvalBatch(
                video=media.video,
                prompts=chunk,
                audio=getattr(media, "audio", None),
            )

    pin = EvalPin(
        checkpoint=str(arguments.checkpoint),
        sampler=settings.sampler,
        steps=settings.steps,
        guidance=settings.guidance,
        negative_prompt=settings.negative_prompt,
        seed=settings.seed,
        frames=settings.frames,
        height=settings.height,
        width=settings.width,
        decoded=settings.decode,
        precision=configuration.inference.dtype,
    )
    return run_eval_suite(
        batches(),
        metrics=settings.metrics,
        pin=pin,
        prompts=prompts,
        decoded=settings.decode,
        gather=gather,
        data_rank=data_rank,
        run_name=configuration.resolved_run_name(),
        step=arguments.step,
    )


def _load_prompts(settings: Any) -> list[str]:
    """Read the evaluation prompt set, or fall back to the built-in list."""
    from avgen.eval import DEFAULT_PROMPTS

    if not settings.prompts_file:
        return list(DEFAULT_PROMPTS)
    source = Path(settings.prompts_file)
    if not source.is_file():
        raise FileNotFoundError(f"eval.prompts_file not found: {source}")
    prompts = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if settings.num_prompts:
        prompts = prompts[: settings.num_prompts]
    if not prompts:
        raise ValueError(f"{source} contains no prompts")
    return prompts
