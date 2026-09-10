"""``avgen train``, ``avgen finetune``, ``avgen rl`` — the three training entry points.

All three share one flow, which is why they share one module: load and validate
the config, record the *resolved* config next to the run, build the parallel
world, and hand off to the subsystem that owns the loop. The differences are in
which subsystem, and in what is required before starting — a fine-tune without a
base checkpoint is not a fine-tune, and RL without a reference policy has
nothing to measure its KL against.

**The resolved config is written before the first step, not after.** A run whose
config exists only in shell history is not reproducible, and the moment to
discover that is not three days in. ``checkpoint.save_config`` controls it and
defaults to on.

Every import of a training subsystem happens inside a handler. These are the
largest subsystems in the package and several of them are still being written;
``avgen --help`` and ``avgen plan`` must not depend on any of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avgen.cli._common import (
    add_config_arguments,
    emit,
    fail,
    human_count,
    load_run_config,
    rule,
)

__all__ = ["add_parser", "run", "run_finetune", "run_rl"]


def _run_loop(configuration: Any) -> None:
    """Build the training stack and hand it to the trainer.

    Kept separate from :func:`run` so that the argument handling above stays
    readable and so a test can drive a real loop without going through argparse.

    Args:
        configuration: The validated run configuration.

    Raises:
        RuntimeError: If a required subsystem is unavailable; the message names
            the contract that declares it.
    """
    from avgen.cli._wiring import (
        build_trainer_config,
        build_training_stack,
        require_subsystem,
    )

    stack = build_training_stack(configuration)
    if configuration.checkpoint.resume:
        load_checkpoint = require_subsystem("avgen.checkpoint", "load")
        load_checkpoint(
            configuration.checkpoint.resume, stack.state, parallel=stack.parallel
        )
    trainer_class = require_subsystem("avgen.train.trainer", "Trainer")
    trainer = trainer_class(
        stack.state,
        stack.objective,
        stack.parallel,
        build_trainer_config(
            configuration,
            accumulation=stack.gradient_accumulation,
            data_world=stack.data_world,
        ),
        logger=stack.logger,
    )
    try:
        trainer.fit(stack.source, total_steps=configuration.train.steps)
    finally:
        # Close the logger even on an interrupt: a jsonl sink with an unflushed
        # tail loses the last metrics, which are the ones you wanted.
        close = getattr(stack.logger, "close", None)
        if callable(close):
            close()


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``train``, ``finetune``, and ``rl``.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The ``train`` parser. The other two are registered as a side effect,
        because they are variants of the same command rather than independent
        ones and keeping them together stops the three drifting apart.
    """
    train = subparsers.add_parser(
        "train",
        help="train a model from a YAML configuration",
        description=(
            "Train a video or audio-video generation model.\n\n"
            "Launch under torchrun for anything beyond one GPU:\n"
            "  torchrun --nproc-per-node 8 -m avgen.cli.main train --config ...\n\n"
            "Any config value is overridable on the command line. Types come "
            "from the schema, so train.lr=1e-4 becomes a float. An unknown key "
            "is an error, never a silent no-op — a typo'd override is a wasted "
            "cluster run."
        ),
        epilog=(
            "examples:\n"
            "  avgen train --config configs/train/smoke_cpu.yaml\n"
            "  avgen train --config configs/train/node_8gpu.yaml train.lr=2e-4 "
            "train.steps=50000\n"
            "  avgen train --config configs/train/multinode_64.yaml "
            "parallel.context=8 --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(train)
    train.add_argument(
        "--resume",
        metavar="PATH",
        default="",
        help=(
            "resume from a checkpoint directory, or 'latest'. Overrides "
            "checkpoint.resume. Restores weights, optimizer, schedule, EMA, and "
            "the data cursor — a resume that restores weights but not the "
            "cursor silently retrains on data it has already seen, and the loss "
            "curve will not tell you."
        ),
    )
    train.set_defaults(handler=run)

    finetune = subparsers.add_parser(
        "finetune",
        help="fine-tune a pretrained checkpoint (LoRA, full, or control tower)",
        description=(
            "Adapt a pretrained model. The mode comes from finetune.mode: "
            "'lora' (adapter), 'full' (every parameter, or a frozen subset), or "
            "'control' (a ControlNet-style side tower).\n\n"
            "finetune.base_checkpoint is required: a fine-tune without a base "
            "is just training, and the two need different learning rates, "
            "schedules, and EMA settings."
        ),
        epilog=(
            "examples:\n"
            "  avgen finetune --config configs/finetune/lora.yaml "
            "finetune.base_checkpoint=runs/base/ckpt-40000\n"
            "  avgen finetune --config configs/finetune/control.yaml "
            "finetune.control_channels=3\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(finetune)
    finetune.set_defaults(handler=run_finetune)

    reinforcement = subparsers.add_parser(
        "rl",
        help="reward or preference post-training (GRPO, DPO)",
        description=(
            "Post-train a model against a reward or a preference dataset.\n\n"
            "'grpo' converts the deterministic sampler to an SDE, rolls out a "
            "group of samples per prompt, and forms a group-relative advantage "
            "with a KL penalty to a frozen reference. 'dpo' optimises a "
            "pairwise preference loss directly.\n\n"
            "Both are dominated by rollout cost: rl.sampler_steps is the first "
            "knob to cut, and rl.kl_coefficient decides whether you get reward "
            "learning or reward hacking."
        ),
        epilog=(
            "examples:\n"
            "  avgen rl --config configs/rl/grpo.yaml rl.group_size=8\n"
            "  avgen rl --config configs/rl/dpo.yaml rl.dpo_beta=0.1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(reinforcement)
    reinforcement.set_defaults(handler=run_rl)
    return train


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every training entry point shares."""
    add_config_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate the config, resolve the parallel plan, print the run "
            "summary, and stop before allocating anything. Costs a second and "
            "catches most reasons a launch dies in its first minute."
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the fully resolved configuration as YAML and exit",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default="",
        help="override output_dir from the config",
    )


def _prepare(arguments: argparse.Namespace) -> tuple[Any, Path]:
    """Load the config, apply CLI-level overrides, and resolve the output root."""
    from dataclasses import replace

    configuration = load_run_config(arguments)
    if arguments.output_dir:
        configuration = replace(configuration, output_dir=arguments.output_dir)
    resume = getattr(arguments, "resume", "")
    if resume:
        configuration = replace(
            configuration,
            checkpoint=replace(configuration.checkpoint, resume=resume),
        )
    return configuration, Path(configuration.output_dir)


def _render_summary(configuration: Any, *, mode: str) -> str:
    """Render the pre-flight summary printed before every run."""
    model = configuration.model
    parallel = configuration.parallel
    bucket = configuration.data.largest_bucket()
    tokens = configuration.sequence_length
    # dp_shard = -1 means "absorb whatever ranks are left", which is what makes
    # one config file work unchanged on 8 or 512 GPUs. Printing the literal -1
    # reads as a misconfiguration, so say what it means.
    degrees = " ".join(
        f"{name}={'auto' if value == -1 else value}"
        for name, value in (
            ("dp_replicate", parallel.dp_replicate),
            ("dp_shard", parallel.dp_shard),
            ("cp", parallel.context),
            ("tp", parallel.tensor),
            ("pp", parallel.pipeline),
        )
        if value != 1
    )
    return "\n".join(
        [
            rule(f"avgen {mode}", character="="),
            f"  run           {configuration.resolved_run_name()}",
            f"  output        {configuration.output_dir}",
            f"  model         {model.name} · "
            f"{human_count(model.estimated_parameters())} params · "
            f"{model.depth} x {model.width} · "
            f"{'AV' if model.has_audio else 'video-only'}",
            f"  data          {configuration.data.source} · bucket "
            f"{bucket.name} ({bucket.frames}x{bucket.height}x{bucket.width}) · "
            f"{tokens:,} tokens/sample",
            f"  parallelism   {degrees or 'single device'} · "
            f"{parallel.precision.param_dtype} params / "
            f"{parallel.precision.reduce_dtype} reduce · "
            f"AC {parallel.activation.mode}",
            f"  optimisation  {configuration.train.optimizer} lr "
            f"{configuration.train.lr:g} · {configuration.train.schedule} · "
            f"{configuration.train.steps:,} steps · global batch "
            f"{configuration.train.global_batch_size}",
            f"  objective     {configuration.train.objective} · timesteps "
            f"{configuration.train.timestep_sampler}",
            f"  telemetry     {', '.join(configuration.telemetry.loggers)} every "
            f"{configuration.telemetry.log_every} steps",
        ]
    )


def _preflight(arguments: argparse.Namespace, *, mode: str) -> tuple[Any, Path] | int:
    """Shared start-up: print, save, and optionally stop before allocating."""
    from avgen.config import save_config

    configuration, output = _prepare(arguments)

    if arguments.print_config:
        import yaml

        from avgen.config.diff import to_mapping

        emit(yaml.safe_dump(to_mapping(configuration), sort_keys=False, width=88))
        return 0

    emit(_render_summary(configuration, mode=mode))

    if arguments.dry_run:
        emit()
        emit("  --dry-run: configuration is valid; stopping before allocation.")
        emit(
            "  Next: 'avgen simulate --config ...' predicts memory and step time "
            "for this exact config."
        )
        return 0

    if configuration.checkpoint.save_config:
        destination = save_config(configuration, output / "config.yaml")
        emit(f"  resolved config written to {destination}")
    return configuration, output


def run(arguments: argparse.Namespace) -> int:
    """Run pretraining.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    prepared = _preflight(arguments, mode="train")
    if isinstance(prepared, int):
        return prepared
    configuration, _ = prepared
    _run_loop(configuration)
    return 0


def run_finetune(arguments: argparse.Namespace) -> int:
    """Run fine-tuning.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    prepared = _preflight(arguments, mode="finetune")
    if isinstance(prepared, int):
        return prepared
    configuration, _ = prepared

    if not configuration.finetune.base_checkpoint:
        return fail(
            "finetune.base_checkpoint is empty. A fine-tune needs a base: set "
            "it in the config or pass "
            "finetune.base_checkpoint=<path> as an override."
        )

    from avgen.cli._wiring import require_subsystem

    apply_finetune = require_subsystem("avgen.finetune.entry", "finetune_from_config")
    apply_finetune(configuration)
    return 0


def run_rl(arguments: argparse.Namespace) -> int:
    """Run reward or preference post-training.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    prepared = _preflight(arguments, mode="rl")
    if isinstance(prepared, int):
        return prepared
    configuration, _ = prepared

    settings = configuration.rl
    if settings.algorithm == "grpo" and not settings.rewards:
        return fail(
            "rl.rewards is empty. GRPO optimises a reward; with none configured "
            "the advantage is identically zero and the run is an expensive "
            "no-op."
        )
    if not settings.reference_checkpoint:
        emit(
            "  note: rl.reference_checkpoint is unset — the KL penalty will be "
            "measured against a snapshot of the starting weights."
        )

    from avgen.cli._wiring import require_subsystem

    post_train = require_subsystem("avgen.rl.entry", "rl_from_config")
    post_train(configuration)
    return 0
