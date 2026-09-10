"""``avgen plan`` — search the parallelism space before spending a cluster-hour.

This command is the project's shop window, and it earns that by being the one
piece of avgen that is useful before you have installed anything else, written
a config, or booked a GPU. Give it a model shape and a world size and it
enumerates every valid factorisation across data, context, tensor, and pipeline
parallelism, prices each with the memory, communication, and compute models, and
ranks what survives. On 512 ranks that is a few thousand candidates and it takes
well under a second on a laptop.

The output is deliberately more than a table. A ranked list tells you *what* to
run; the sections underneath tell you *why* it won, which is the part that
transfers to the next model. The three things worth reading every time:

* **dominant memory term** — the thing to attack when nothing fits. If it is
  ``activation``, more context parallelism helps and more FSDP does not.
* **bottleneck** — the collective costing the most time. For video it is
  usually ``context_parallel.ring_kv``, which is a fact about attention rather
  than about your code.
* **scaling efficiency** — the fraction of ideal throughput left after
  communication. Below 0.8 a different factorisation of the same world size
  will beat this one.

Every number here is **predicted**. The estimator is exact about shapes,
sharding, and collective sizes, and approximate about time, because it does not
model your fabric's congestion or your scheduler's placement. Calibrate it once
against a real run and every projection downstream improves.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from typing import Any

from avgen.cli._common import (
    emit,
    fail,
    human_count,
    human_seconds,
    rule,
    table,
)

__all__ = ["add_parser", "run"]

#: How many candidates the ranked table shows by default. Beyond about ten the
#: remaining plans are permutations of the same idea and the table stops
#: informing.
DEFAULT_TOP_K = 8


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``plan`` subcommand.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "plan",
        help="rank parallelism plans for a model shape and world size",
        description=(
            "Enumerate every valid way to factor a world size across data, "
            "context, tensor, and pipeline parallelism; price each one for "
            "memory, communication, and compute; and rank what fits.\n\n"
            "Needs no config file, no GPU, and no installed model. This is the "
            "command to run before booking the cluster, not after."
        ),
        epilog=(
            "examples:\n"
            "  # a 2B video DiT on 512 H100s at 64k tokens\n"
            "  avgen plan --world-size 512 --seq-len 65536 --params 2e9 "
            "--depth 32 --width 2560\n\n"
            "  # the same model on Blackwell, with a fixed global batch\n"
            "  avgen plan --world-size 256 --seq-len 65536 --params 2e9 "
            "--depth 32 --width 2560 --gpu b200 --global-batch 256\n\n"
            "  # allow pipeline parallelism and emit machine-readable output\n"
            "  avgen plan --world-size 1024 --seq-len 131072 --params 14e9 "
            "--depth 48 --width 5120 --max-pipeline 4 --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    shape = parser.add_argument_group("model shape")
    shape.add_argument(
        "--params",
        type=float,
        default=2e9,
        metavar="N",
        help="parameter count, e.g. 2e9 (default: %(default)s)",
    )
    shape.add_argument(
        "--depth",
        type=int,
        default=32,
        help="transformer blocks (default: %(default)s)",
    )
    shape.add_argument(
        "--width", type=int, default=2560, help="hidden width (default: %(default)s)"
    )
    shape.add_argument(
        "--heads",
        type=int,
        default=0,
        help=(
            "attention heads; 0 derives them at 128 dims per head "
            "(default: %(default)s)"
        ),
    )
    shape.add_argument(
        "--mlp-ratio",
        type=int,
        default=4,
        help="feed-forward expansion factor (default: %(default)s)",
    )
    shape.add_argument(
        "--seq-len",
        type=int,
        default=65536,
        metavar="L",
        help=(
            "tokens per sample before context-parallel sharding. For video this "
            "is the number that decides everything (default: %(default)s)"
        ),
    )
    shape.add_argument(
        "--text-tokens",
        type=int,
        default=226,
        help=(
            "cross-attention context length; 0 for no cross-attention "
            "(default: %(default)s)"
        ),
    )
    shape.add_argument(
        "--micro-batch",
        type=int,
        default=1,
        help="samples per rank per microbatch (default: %(default)s)",
    )

    cluster = parser.add_argument_group("cluster")
    cluster.add_argument(
        "--world-size",
        type=int,
        required=True,
        metavar="N",
        help="total ranks available",
    )
    cluster.add_argument(
        "--gpu",
        default="h100",
        choices=("a100", "h100", "h200", "b200"),
        help=(
            "device profile; also selects the intra-node fabric (default: %(default)s)"
        ),
    )
    cluster.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help=(
            "ranks sharing the fast fabric. Tensor parallelism is never allowed "
            "to exceed this (default: %(default)s)"
        ),
    )
    cluster.add_argument(
        "--global-batch",
        type=int,
        default=0,
        metavar="B",
        help=(
            "samples per optimizer step. When set, plans that cannot factor it "
            "are rejected rather than silently changing your effective batch "
            "size (default: unconstrained)"
        ),
    )

    search = parser.add_argument_group("search bounds")
    search.add_argument("--max-tensor", type=int, default=8, help="cap on TP degree")
    search.add_argument("--max-context", type=int, default=32, help="cap on CP degree")
    search.add_argument(
        "--max-pipeline",
        type=int,
        default=1,
        help=(
            "cap on PP degree. Left at 1 because pipeline is the last axis to "
            "reach for: its bubble costs real throughput and its benefit only "
            "appears when depth alone is the constraint"
        ),
    )
    search.add_argument(
        "--min-blocks-per-stage",
        type=int,
        default=4,
        help="refuse pipeline splits thinner than this (default: %(default)s)",
    )
    search.add_argument(
        "--no-hsdp",
        action="store_true",
        help="exclude replicated-over-sharded (HSDP) data parallelism",
    )
    search.add_argument(
        "--achieved-fraction",
        type=float,
        default=0.45,
        help=(
            "fraction of peak FLOPs assumed reachable. 0.45 is realistic for a "
            "well-tuned large transformer; calibrate from a real run "
            "(default: %(default)s)"
        ),
    )
    search.add_argument(
        "--headroom",
        type=float,
        default=0.10,
        help=(
            "device-memory fraction kept free for allocator fragmentation and "
            "NCCL buffers (default: %(default)s)"
        ),
    )
    search.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="candidates to show (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the ranked candidates as JSON instead of a table",
    )
    parser.set_defaults(handler=run)
    return parser


def run(arguments: argparse.Namespace) -> int:
    """Search and rank parallelism plans.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    # Imported here rather than at module scope so ``avgen --help`` does not
    # pay for torch. The simulate package pulls it in transitively.
    from avgen.config.resolve import build_accelerator, build_interconnects
    from avgen.simulate.compute import transformer_flops
    from avgen.simulate.memory import ModelShape
    from avgen.simulate.plan import SearchSpace, search_parallel_plan

    if arguments.world_size < 1:
        return fail(f"--world-size must be positive; got {arguments.world_size}")
    if arguments.depth < 1 or arguments.width < 1:
        return fail("--depth and --width must be positive")

    heads = arguments.heads or max(1, arguments.width // 128)
    if arguments.width % heads != 0:
        return fail(
            f"--heads {heads} does not divide --width {arguments.width}; "
            f"the remainder is {arguments.width % heads}"
        )

    shape = ModelShape(
        parameters=int(arguments.params),
        depth=arguments.depth,
        width=arguments.width,
        sequence_length=arguments.seq_len,
        micro_batch_size=arguments.micro_batch,
        mlp_ratio=arguments.mlp_ratio,
        num_heads=heads,
        text_tokens=arguments.text_tokens,
    )
    accelerator = build_accelerator(arguments.gpu)
    intra_node, inter_node = build_interconnects(arguments.gpu)
    space = SearchSpace(
        world_size=arguments.world_size,
        gpus_per_node=arguments.gpus_per_node,
        max_tensor=arguments.max_tensor,
        max_context=arguments.max_context,
        max_pipeline=arguments.max_pipeline,
        allow_hsdp=not arguments.no_hsdp,
        global_batch_size=arguments.global_batch or None,
        min_blocks_per_stage=arguments.min_blocks_per_stage,
    )

    started = time.perf_counter()
    # Ask for far more than we display so the "how many fit" line is honest and
    # the caveat section can reason about the tail.
    candidates = search_parallel_plan(
        shape,
        space,
        accelerator=accelerator,
        intra_node=intra_node,
        inter_node=inter_node,
        achieved_fraction=arguments.achieved_fraction,
        headroom=arguments.headroom,
        top_k=10_000,
    )
    elapsed = time.perf_counter() - started

    if arguments.json:
        payload = {
            "shape": {
                "parameters": shape.parameters,
                "depth": shape.depth,
                "width": shape.width,
                "num_heads": shape.num_heads,
                "sequence_length": shape.sequence_length,
                "micro_batch_size": shape.micro_batch_size,
                "text_tokens": shape.text_tokens,
            },
            "accelerator": accelerator.name,
            "world_size": arguments.world_size,
            "search_seconds": round(elapsed, 4),
            "fitting_plans": len(candidates),
            "candidates": [
                item.summary(accelerator)
                for item in candidates[: max(1, arguments.top_k)]
            ],
        }
        emit(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    flops = transformer_flops(shape)
    emit(_render_header(arguments, shape, accelerator, intra_node, inter_node, flops))
    emit(
        f"  search     {len(candidates):,} plans fit in memory, priced in "
        f"{human_seconds(elapsed)}"
    )
    emit()

    if not candidates:
        emit(_render_nothing_fits(shape, space, accelerator, arguments))
        return 1

    shown = candidates[: max(1, arguments.top_k)]
    emit(rule("RANKED BY PREDICTED THROUGHPUT"))
    emit(_render_table(shown, accelerator))
    emit()
    emit(_render_best(shown[0], accelerator, arguments))
    emit()
    emit(_render_analysis(shown, candidates, flops, shape))
    emit()
    emit(_render_caveats(accelerator, intra_node, inter_node, arguments))
    return 0


def _render_header(
    arguments: argparse.Namespace,
    shape: Any,
    accelerator: Any,
    intra_node: Any,
    inter_node: Any,
    flops: dict[str, float],
) -> str:
    """Render the block describing what is being planned."""
    nodes = max(1, arguments.world_size // max(1, arguments.gpus_per_node))
    head_dim = shape.width // shape.num_heads
    peak = f"{accelerator.bf16_tflops:.0f} TF bf16"
    if accelerator.fp8_tflops:
        peak += f" / {accelerator.fp8_tflops:.0f} TF fp8"
    batch = (
        f"global {arguments.global_batch}"
        if arguments.global_batch
        else "global unconstrained"
    )
    return "\n".join(
        [
            rule("avgen plan — parallelism search", character="="),
            "",
            f"  model      {human_count(shape.parameters)} parameters · "
            f"{shape.depth} blocks x {shape.width} wide · "
            f"{shape.num_heads} heads ({head_dim}/head) · mlp {shape.mlp_ratio}x",
            f"  sequence   {shape.sequence_length:,} tokens/sample · "
            f"attention is {flops['attention_fraction']:.0%} of forward FLOPs · "
            f"{human_count(flops['total'] / shape.micro_batch_size)} FLOPs/sample/step",
            f"  hardware   {arguments.world_size:,} x {accelerator.name} · "
            f"{accelerator.memory_gib:.0f} GiB · {peak} · "
            f"{nodes} node(s) x {arguments.gpus_per_node}",
            f"  fabric     {intra_node.name} {intra_node.peak_gbps:.0f} GB/s "
            f"intra-node · {inter_node.name} {inter_node.peak_gbps:.0f} GB/s "
            "inter-node",
            f"  batch      {batch} · micro {shape.micro_batch_size}/rank",
        ]
    )


def _render_table(candidates: list[Any], accelerator: Any) -> str:
    """Render the ranked candidate table."""
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        summary = candidate.summary(accelerator)
        rows.append(
            {
                "rank": str(index),
                "plan": candidate.dims.describe().replace("world=", "w="),
                "ac": summary["activation_checkpoint"],
                "mem": f"{summary['memory_gib']:.1f}",
                "free": f"{summary['memory_headroom_gib']:.1f}",
                "step": human_seconds(summary["step_seconds"]),
                "samples": f"{summary['samples_per_second']:.1f}",
                "mfu": f"{summary['mfu']:.1%}",
                "scaling": f"{summary['scaling_efficiency']:.0%}",
                "comm": f"{summary['comm_gib_per_step']:.2f}",
                "bottleneck": summary["bottleneck"],
                "memterm": summary["dominant_memory"].removesuffix("_bytes"),
            }
        )
    return table(
        [
            ("rank", "#"),
            ("plan", "plan"),
            ("ac", "checkpoint"),
            ("mem", "GiB"),
            ("free", "free"),
            ("step", "step"),
            ("samples", "samp/s"),
            ("mfu", "MFU"),
            ("scaling", "scale"),
            ("comm", "GiB/step"),
            ("bottleneck", "bottleneck"),
            ("memterm", "top memory"),
        ],
        rows,
        aligns=["r", "l", "l", "r", "r", "r", "r", "r", "r", "r", "l", "l"],
    )


def _override_string(candidate: Any) -> str:
    """Render the plan as the exact ``avgen train`` overrides that select it."""
    dims = candidate.dims
    parts = [f"parallel.dp_shard={dims.dp_shard}"]
    if dims.dp_replicate > 1:
        parts.append(f"parallel.dp_replicate={dims.dp_replicate}")
    if dims.context > 1:
        parts.append(f"parallel.context={dims.context}")
    if dims.tensor > 1:
        parts.append(f"parallel.tensor={dims.tensor}")
    if dims.pipeline > 1:
        parts.append(f"parallel.pipeline={dims.pipeline}")
    parts.append(f"parallel.activation.mode={candidate.activation_checkpoint.mode}")
    if candidate.activation_checkpoint.mode == "selective_op":
        parts.append(
            "parallel.activation.save_op_frequency="
            f"{candidate.activation_checkpoint.save_op_frequency}"
        )
    return " ".join(parts)


def _render_best(
    candidate: Any,
    accelerator: Any,
    arguments: argparse.Namespace,
) -> str:
    """Render the detailed breakdown of the winning plan."""
    memory = candidate.memory
    communication = candidate.communication
    free = accelerator.memory_gib - memory.total_gib
    breakdown = " · ".join(
        f"{name.removesuffix('_bytes')} {value:.1f}"
        for name, value in memory.breakdown_gib().items()
        if value >= 0.05
    )
    comm_lines = [
        f"      {label:<32} {seconds * 1e3:8.2f} ms"
        for label, seconds in communication.by_label().items()
    ]
    tokens_per_second = (
        candidate.samples_per_second * arguments.seq_len
        if candidate.samples_per_second
        else 0.0
    )
    lines = [
        rule("BEST PLAN", character="="),
        f"  {candidate.dims.describe()}   "
        f"activation checkpoint: {candidate.activation_checkpoint.mode}",
        "",
        f"  memory       {memory.total_gib:.1f} GiB of "
        f"{accelerator.memory_gib:.0f} GiB per rank "
        f"({free / accelerator.memory_gib:.0%} free)",
        f"      {breakdown}",
        "",
        f"  time         step {human_seconds(candidate.step_seconds)} = "
        f"compute {human_seconds(candidate.compute.seconds)} + exposed comm "
        f"{human_seconds(communication.exposed_seconds)}"
        + (
            f" x {candidate.gradient_accumulation} microbatches"
            if candidate.gradient_accumulation > 1
            else ""
        ),
        f"  communication {communication.total_wire_gib:.2f} GiB/step on the wire",
        *comm_lines,
        "",
        f"  utilisation  MFU {candidate.mfu(accelerator):.1%} · "
        f"scaling efficiency {communication.scaling_efficiency:.1%} · "
        f"recompute overhead {candidate.compute.recompute_overhead:.1%}",
        f"  throughput   {candidate.samples_per_second:.1f} samples/s · "
        f"{human_count(tokens_per_second)} tokens/s",
        "",
        "  select it with:",
        "      avgen train --config <your-config>.yaml \\",
        f"          {_override_string(candidate)}",
    ]
    return "\n".join(lines)


def _render_analysis(
    shown: list[Any],
    fitting: list[Any],
    flops: dict[str, float],
    shape: Any,
) -> str:
    """Render the "what actually moves this answer" section.

    Args:
        shown: The candidates displayed in the table.
        fitting: Every candidate that fits, which is what the spread is
            computed over — quoting a spread across only the displayed rows
            would understate it by construction.
        flops: Output of :func:`~avgen.simulate.compute.transformer_flops`.
        shape: The model geometry.

    Returns:
        A wrapped, bulleted analysis.
    """
    best = shown[0]
    notes: list[str] = []

    if flops["attention_fraction"] > 0.5:
        notes.append(
            f"Attention is {flops['attention_fraction']:.0%} of forward FLOPs at "
            f"{shape.sequence_length:,} tokens. The language-model shortcut "
            "'6 * params * tokens' would understate this run's work several-fold "
            "and report an impossibly low MFU. Sequence length, not parameter "
            "count, is the cost driver here."
        )
    if best.dims.cp_enabled:
        notes.append(
            f"Context parallelism at cp={best.dims.context} is doing the work: it "
            "is the only axis that reduces per-rank sequence length, and both "
            "activation memory and attention cost scale with that."
        )
    else:
        notes.append(
            "The winner uses no context parallelism, which means the sequence "
            "already fits comfortably. Expect that to change with resolution or "
            "clip length — attention cost grows quadratically in both."
        )
    if best.memory.dominant_term() == "activation_bytes":
        notes.append(
            "Activations dominate memory, not parameters. More FSDP sharding "
            "will not help; more context parallelism, a shorter clip, or a more "
            "aggressive checkpointing policy will."
        )
    elif best.memory.dominant_term() == "optimizer_bytes":
        notes.append(
            "Optimizer state dominates memory. AdamW in mixed precision costs 12 "
            "bytes per parameter — an fp32 master copy plus two fp32 moments, "
            "not the 8 people usually assume — so sharding optimizer state "
            "across dp_shard is what buys the room."
        )
    if best.communication.scaling_efficiency < 0.8:
        notes.append(
            f"Scaling efficiency is {best.communication.scaling_efficiency:.0%}: "
            f"{best.communication.bottleneck()} is not hidden behind compute. A "
            "different factorisation of the same world size is likely to beat "
            "this one; try raising --max-context or lowering --max-tensor."
        )
    if len(fitting) > 1:
        spread = fitting[0].samples_per_second / max(
            1e-9, fitting[-1].samples_per_second
        )
        lightest = min(item.memory.total_gib for item in fitting)
        heaviest = max(item.memory.total_gib for item in fitting)
        if spread >= 1.15:
            notes.append(
                f"The best and worst *fitting* plans differ by {spread:.1f}x in "
                "predicted throughput. That spread is why this search exists: "
                "picking by intuition leaves most of a cluster on the floor, and "
                "finding out costs one job launch per guess."
            )
        else:
            # Throughput is near-flat across plans whenever communication is
            # fully hidden: model-parallel degrees divide both the per-rank work
            # and the data-parallel width, and the two cancel. Saying so is more
            # useful than reporting a 1.0x spread as if it were a finding.
            notes.append(
                f"Predicted throughput is flat across all {len(fitting)} "
                f"fitting plans (spread {spread:.2f}x): communication is fully "
                "hidden, so every factorisation does the same arithmetic per "
                "rank. Choose on memory headroom and scaling efficiency instead "
                f"— per-rank memory ranges {lightest:.1f}-{heaviest:.1f} GiB "
                "across these plans, and headroom is what absorbs a longer clip "
                "or a bigger bucket later."
            )
    modes = {item.activation_checkpoint.mode for item in shown[:5]}
    if "none" in modes:
        notes.append(
            "Some plans fit with no activation checkpointing at all. Prefer "
            "those: recomputation is real work that lands in the hardware-FLOPs "
            "denominator and never in the model-FLOPs numerator."
        )
    wrapped: list[str] = []
    for note in notes:
        wrapped.extend(
            textwrap.wrap(
                note,
                width=78,
                initial_indent="  - ",
                subsequent_indent="    ",
            )
        )
        wrapped.append("")
    return "\n".join([rule("WHAT MOVES THIS ANSWER"), *wrapped]).rstrip()


def _render_caveats(
    accelerator: Any,
    intra_node: Any,
    inter_node: Any,
    arguments: argparse.Namespace,
) -> str:
    """Render the honesty section."""
    return "\n".join(
        [
            rule("CAVEATS"),
            "  These are predictions, not measurements. The estimator is exact "
            "about shapes,",
            "  sharding, and collective sizes, and approximate about time: it "
            "models neither",
            "  fabric congestion, nor scheduler placement, nor your neighbours' "
            "traffic.",
            "",
            f"  - Peak FLOPs for {accelerator.name} are published dense figures, "
            "not your silicon.",
            f"  - Fabric bandwidths ({intra_node.name}, {inter_node.name}) are "
            "plausible defaults.",
            "    Calibrate them once with nccl-tests and "
            "avgen.simulate.comms.calibrate_from_busbw",
            "    and every projection downstream becomes a projection instead of "
            "a guess.",
            f"  - Achieved-FLOPs fraction is assumed at "
            f"{arguments.achieved_fraction:.2f}; measure your own.",
            "  - Run 'avgen simulate --config <yours>' to price your real "
            "configured model,",
            "    including a fake-process-group check that the plan actually "
            "applies at scale.",
        ]
    )


def _render_nothing_fits(
    shape: Any,
    space: Any,
    accelerator: Any,
    arguments: argparse.Namespace,
) -> str:
    """Explain a failed search with the arithmetic that caused it."""
    from avgen.parallel.activation import ActivationCheckpointConfig
    from avgen.parallel.dims import ParallelDims
    from avgen.simulate.memory import estimate_memory

    lines = [
        rule("NO CONFIGURATION FITS", character="="),
        "",
        "  Every valid factorisation exceeds device memory. This is an "
        "arithmetic fact,",
        "  not a tuning problem: no checkpointing policy recovers it.",
        "",
    ]

    # Price the most aggressive plan the search bounds allow, so the message
    # says how far off the answer is rather than only that it failed.
    best_context = min(space.max_context, space.world_size)
    for context in range(best_context, 0, -1):
        if space.world_size % context or shape.sequence_length % context:
            continue
        try:
            dims = ParallelDims(world_size=space.world_size, context=context)
        except ValueError:
            continue
        estimate = estimate_memory(
            shape, dims, activation_checkpoint=ActivationCheckpointConfig(mode="full")
        )
        deficit = estimate.total_gib - accelerator.memory_gib
        lines += [
            f"  Most aggressive plan in bounds: {dims.describe()} with full "
            "checkpointing",
            f"    needs {estimate.total_gib:.1f} GiB against "
            f"{accelerator.memory_gib:.0f} GiB available "
            f"({deficit:+.1f} GiB).",
            f"    Largest term: {estimate.dominant_term().removesuffix('_bytes')}.",
            "",
        ]
        break

    lines += [
        "  Options, in order of how much they help:",
        "    1. Raise --max-context (needs more ranks). Sequence length is "
        "almost always",
        "       the binding constraint for video, and context parallelism is "
        "the only axis",
        "       that reduces it.",
        f"    2. Shorten the clip or lower the resolution: --seq-len is "
        f"currently {shape.sequence_length:,}.",
        "    3. Use a device with more memory: --gpu h200 (141 GiB) or "
        "--gpu b200 (192 GiB).",
        "    4. Reduce --depth or --width.",
        "",
        f"  Raising --headroom below {arguments.headroom} is not on this list. "
        "The margin covers",
        "  allocator fragmentation and NCCL buffers, and a job that peaks at "
        "99% of device",
        "  memory OOMs on the one step whose bucket is slightly larger.",
    ]
    return "\n".join(lines)
