"""``avgen simulate`` — price the real configured model on a world that does not exist.

Where ``avgen plan`` searches a space of hypothetical shapes, this command takes
*your* config — the one you are about to launch — and reports what it will do at
a world size you do not have. It answers three questions that otherwise cost a
job launch each:

1. **Does it fit?** Per-rank memory, broken down by term, against the device.
2. **How fast?** Predicted step time, MFU, and where the time goes.
3. **Is the plan shaped right?** With ``--topology`` it builds a real
   ``DeviceMesh`` over a ``FakeProcessGroup``, so the mesh ordering, the
   flattened sub-meshes, and this rank's data and sequence coordinates are the
   real ones — at 1024 ranks, on one machine, in a second.

Point three catches a class of bug that no amount of small-scale testing finds:
a mesh dimension ordered so tensor-parallel traffic crosses nodes, a
context-parallel shard index that does not vary the way the loader assumes, a
data rank that collides. Those produce no error at any scale. They produce a
slow job, or a job whose effective batch size is a fraction of what the config
says.

**What a simulation cannot tell you**: whether the model learns. Fake
collectives return uninitialised memory, so a simulated loss is meaningless. Use
this for "does the plan fit and is it shaped right", never for numerics.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from avgen.cli._common import (
    add_config_arguments,
    emit,
    fail,
    human_count,
    load_run_config,
    rule,
)

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``simulate`` subcommand.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "simulate",
        help="predict memory, time, and collectives for a configured run",
        description=(
            "Price the model in your config at a world size you do not have. "
            "Runs on CPU in about a second and needs no GPU.\n\n"
            "Reports per-rank memory by term, predicted step time and MFU, the "
            "collectives each step issues and what they cost, and (with "
            "--topology) the real device mesh built over a fake process group."
        ),
        epilog=(
            "examples:\n"
            "  avgen simulate --config configs/train/multinode_64.yaml\n"
            "  avgen simulate --config configs/train/node_8gpu.yaml "
            "--world-size 1024 parallel.context=8\n"
            "  avgen simulate --config configs/train/node_8gpu.yaml "
            "--topology --rank 137\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--world-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "ranks to simulate. Defaults to the product of the config's "
            "parallelism degrees, or 1 when they are all unset."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        metavar="R",
        help=(
            "rank to impersonate for --topology. Rank 0 is the usual choice; a "
            "middle rank is useful when checking pipeline stage balance "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--gpu",
        default="h100",
        choices=("a100", "h100", "h200", "b200"),
        help="device profile to price against (default: %(default)s)",
    )
    parser.add_argument(
        "--achieved-fraction",
        type=float,
        default=0.45,
        help="fraction of peak FLOPs assumed reachable (default: %(default)s)",
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help=(
            "additionally build the real DeviceMesh over a FakeProcessGroup and "
            "report mesh ordering plus this rank's data and sequence "
            "coordinates. Slower (it initialises a process group) and the only "
            "way to catch a mis-ordered mesh before the cluster does."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON instead of a rendered page",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default="",
        help=(
            "write the JSON report here as well. Commit one as a baseline and "
            "SimulationReport.assert_no_regression turns parallelism "
            "performance into a CI test."
        ),
    )
    parser.set_defaults(handler=run)
    return parser


def _resolved_world_size(configuration: Any, requested: int) -> tuple[int, str]:
    """Return the world size to simulate, and a note when it had to be guessed.

    A config with ``dp_shard: -1`` deliberately does not pin a world size — that
    is the whole point of ``-1``, and it is what lets one file run unchanged on
    8 or 512 ranks. So the config alone underdetermines the answer, and
    defaulting silently to ``dp_shard=1`` would report a 256-rank simulation for
    a file whose comments describe 1024 ranks.

    Args:
        configuration: The run configuration.
        requested: ``--world-size``, or 0 when it was not given.

    Returns:
        The world size, and a note to print (empty when nothing was inferred).
    """
    if requested:
        return requested, ""
    spec = configuration.parallel
    if spec.dp_shard == -1:
        product = spec.dp_replicate * spec.tensor * spec.context * spec.pipeline
        return product, (
            f"  note: parallel.dp_shard is -1, so this config does not pin a "
            f"world size.\n"
            f"        Simulating {product} ranks (dp_shard=1). Pass "
            f"--world-size N to simulate the\n"
            f"        scale you will actually launch at; dp_shard then absorbs "
            "the remainder."
        )
    return (
        max(
            1,
            spec.dp_replicate
            * spec.dp_shard
            * spec.tensor
            * spec.context
            * spec.pipeline,
        ),
        "",
    )


def run(arguments: argparse.Namespace) -> int:
    """Simulate the configured run and print the report.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        ``0`` when the configuration is predicted to fit, ``1`` when it is not.
        The exit code is load-bearing: a CI job can gate a config change on it.
    """
    from avgen.config.resolve import (
        build_accelerator,
        build_activation_checkpoint,
        build_interconnects,
        build_model_shape,
        build_parallel_dims,
    )
    from avgen.simulate.report import simulate_config

    configuration = load_run_config(arguments)
    world_size, note = _resolved_world_size(configuration, arguments.world_size)
    if note and not arguments.json:
        emit(note)
        emit()

    try:
        dims = build_parallel_dims(configuration, world_size=world_size)
    except ValueError as error:
        return fail(
            f"{error}\n"
            f"       Run 'avgen plan --world-size {world_size} "
            f"--seq-len {configuration.sequence_length} "
            f"--params {configuration.model.estimated_parameters()} "
            f"--depth {configuration.model.depth} "
            f"--width {configuration.model.width}' to see the factorisations "
            "that do work."
        )

    shape = build_model_shape(configuration)
    accelerator = build_accelerator(arguments.gpu)
    intra_node, inter_node = build_interconnects(arguments.gpu)

    try:
        accumulation = dims.gradient_accumulation_for(
            global_batch_size=configuration.train.global_batch_size,
            local_batch_size=configuration.micro_batch_size,
        )
    except ValueError as error:
        return fail(str(error))

    report = simulate_config(
        shape,
        dims,
        accelerator=accelerator,
        activation_checkpoint=build_activation_checkpoint(configuration),
        intra_node=intra_node,
        inter_node=inter_node,
        achieved_fraction=arguments.achieved_fraction,
        gradient_accumulation=accumulation,
    )

    payload = report.to_dict()
    payload["run_name"] = configuration.resolved_run_name()
    payload["model_name"] = configuration.model.name
    payload["bucket"] = configuration.data.largest_bucket().name

    if arguments.topology:
        payload["topology"] = _describe_topology(dims, arguments.rank)

    if arguments.save:
        report.save(arguments.save)

    if arguments.json:
        emit(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.fits else 1

    emit(report.render())
    emit()
    emit(_render_context(configuration, shape, accumulation))
    if arguments.topology:
        emit()
        emit(_render_topology(payload["topology"]))
    if arguments.save:
        emit()
        emit(f"  report written to {arguments.save}")
    return 0 if report.fits else 1


def _render_context(configuration: Any, shape: Any, accumulation: int) -> str:
    """Render the parts of the config the estimator does not print itself."""
    tokens_per_step = shape.sequence_length * configuration.train.global_batch_size
    data_parallel = configuration.train.global_batch_size // max(
        1, shape.micro_batch_size * accumulation
    )
    total_tokens = tokens_per_step * configuration.train.steps
    return "\n".join(
        [
            rule("RUN"),
            f"  config        {configuration.resolved_run_name()} "
            f"({configuration.model.name})",
            f"  bucket        {configuration.data.largest_bucket().name} · "
            f"{shape.sequence_length:,} tokens/sample",
            f"  schedule      {configuration.train.steps:,} steps · "
            f"{configuration.train.schedule} · lr {configuration.train.lr:g} · "
            f"warmup {configuration.train.warmup_steps:,}",
            f"  batch         global {configuration.train.global_batch_size} = "
            f"micro {shape.micro_batch_size} x accum {accumulation} x dp "
            f"{data_parallel}",
            f"  token budget  {human_count(total_tokens)} generative tokens over "
            "the full schedule",
        ]
    )


def _describe_topology(dims: Any, rank: int) -> dict[str, Any]:
    """Build the real mesh over a fake process group and describe it.

    Isolated in its own function because it is the only part of this command
    that initialises a process group, and it must be able to fail without
    taking the memory and timing report with it — those are useful even when
    the mesh cannot be built.
    """
    from avgen.simulate.world import fake_world

    if not 0 <= rank < dims.world_size:
        return {
            "error": (
                f"--rank {rank} is outside [0, {dims.world_size}); nothing to "
                "impersonate"
            )
        }
    try:
        with fake_world(dims, rank=rank) as world:
            return dict(world.describe())
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _render_topology(topology: dict[str, Any]) -> str:
    """Render the simulated mesh description."""
    if "error" in topology:
        return "\n".join(
            [
                rule("TOPOLOGY"),
                f"  could not build the mesh: {topology['error']}",
                "  The memory and timing figures above are unaffected.",
            ]
        )
    order = " -> ".join(topology.get("mesh_order", []))
    dimensions = " · ".join(
        f"{name}={size}" for name, size in topology.get("mesh_dims", {}).items()
    )
    return "\n".join(
        [
            rule("TOPOLOGY  (real DeviceMesh over a FakeProcessGroup)"),
            f"  world         {topology['world_size']} ranks, "
            f"impersonating rank {topology['rank']}",
            f"  mesh order    {order or '(single device)'}",
            "                outermost first; the LAST dimension varies "
            "fastest, so tp ranks",
            "                land adjacent inside one NVLink domain and pp "
            "ranks land furthest",
            "                apart. Getting this backwards costs throughput and "
            "raises no error.",
            f"  dimensions    {dimensions or '(none above 1)'}",
            f"  data          rank {topology['data_rank']} of "
            f"{topology['data_world']} — every rank sharing this index must "
            "receive",
            "                byte-identical batches and draw identical noise",
            f"  sequence      shard {topology['sequence_shard']} of "
            f"{topology['sequence_shards']}",
            f"  model replica spread across {topology['model_shard_size']} ranks",
        ]
    )
