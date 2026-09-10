"""``avgen generate`` — sample from a checkpoint.

The flags mirror the pinned fields of an :class:`~avgen.eval.EvalPin` exactly,
and that is not a coincidence. Steps, guidance, sampler, seed, and the negative
prompt are the settings that decide what comes out, and the settings people
forget to record. Making them the visible surface of the command is the cheapest
way to keep them in view; ``--print-settings`` renders them in the same form the
evaluation report pins, so a sample and a metric can be traced to the same
configuration.

Two defaults worth explaining:

* ``--seed`` defaults to ``-1``, meaning "draw a fresh one and print it". A
  fixed default seed makes every user's first sample identical, which reads as
  determinism and is actually a hidden constant. Printing the drawn seed makes
  a good sample reproducible.
* ``--guidance`` defaults to the config's value rather than to a constant.
  Guidance interacts with the timestep shift and the sampler, so a house default
  belongs in the config next to those, not in the argument parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avgen.cli._common import add_traceback_argument, emit, fail, rule

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``generate`` subcommand.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "generate",
        help="sample video (and audio) from a checkpoint",
        description=(
            "Generate media from a trained checkpoint.\n\n"
            "The flags here are exactly the settings an evaluation report pins, "
            "because they are exactly the settings that change the output. Two "
            "samples produced at different --steps or --guidance are not "
            "comparable, and neither are the metrics computed from them."
        ),
        epilog=(
            "examples:\n"
            "  avgen generate --checkpoint runs/base/ckpt-40000 "
            '--prompt "a red balloon rising over a field"\n'
            "  avgen generate --checkpoint runs/base/ckpt-40000 "
            '--prompt "waves on a shore" --steps 50 --guidance 7.5 --seed 1234\n'
            "  avgen generate --checkpoint ckpt --prompt-file prompts.txt "
            "--output samples/ --frames 33\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="checkpoint directory or exported safetensors file",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="",
        help=(
            "run configuration. Defaults to config.yaml inside the checkpoint "
            "directory, which avgen writes there for exactly this reason."
        ),
    )

    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", metavar="TEXT", help="a single prompt")
    prompts.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="newline-delimited prompt file; blank lines and # comments ignored",
    )

    sampling = parser.add_argument_group("sampling (these are the pinned settings)")
    sampling.add_argument(
        "--negative-prompt",
        metavar="TEXT",
        default=None,
        help=(
            "prompt for the unconditional branch. An empty string and a tuned "
            "negative prompt are different experiments; this is recorded in "
            "the pin."
        ),
    )
    sampling.add_argument(
        "--steps",
        type=int,
        default=0,
        help="denoising steps; 0 uses inference.steps from the config",
    )
    sampling.add_argument(
        "--guidance",
        type=float,
        default=None,
        help=(
            "classifier-free guidance scale. 1.0 disables guidance and roughly "
            "halves the cost, because the unconditional branch stops being "
            "evaluated. Defaults to inference.guidance."
        ),
    )
    sampling.add_argument(
        "--guidance-rescale",
        type=float,
        default=None,
        help=(
            "CFG-rescale factor, which counteracts the over-saturation high "
            "guidance produces. Defaults to inference.guidance_rescale."
        ),
    )
    sampling.add_argument(
        "--sampler",
        default="",
        choices=("", "euler", "heun", "dpmpp_2m", "res_multistep"),
        help="sampler; empty uses inference.sampler",
    )
    sampling.add_argument(
        "--seed",
        type=int,
        default=-1,
        help=(
            "sampling seed; -1 draws a fresh one and prints it so the sample "
            "stays reproducible (default: %(default)s)"
        ),
    )

    geometry = parser.add_argument_group("output geometry")
    geometry.add_argument("--frames", type=int, default=0, help="latent frames")
    geometry.add_argument("--height", type=int, default=0, help="latent rows")
    geometry.add_argument("--width", type=int, default=0, help="latent columns")
    geometry.add_argument(
        "--batch-size", type=int, default=0, help="prompts per forward pass"
    )
    geometry.add_argument(
        "--output",
        metavar="PATH",
        default="",
        help="output file or directory; empty uses inference.output",
    )
    parser.add_argument(
        "--print-settings",
        action="store_true",
        help="print the resolved sampling settings and exit without generating",
    )
    add_traceback_argument(parser)
    parser.set_defaults(handler=run)
    return parser


def _read_prompts(arguments: argparse.Namespace) -> list[str]:
    """Return the prompt list from ``--prompt`` or ``--prompt-file``."""
    if arguments.prompt:
        return [arguments.prompt]
    source = Path(arguments.prompt_file)
    if not source.is_file():
        raise FileNotFoundError(f"prompt file not found: {source}")
    prompts = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"prompt file {source} contains no prompts")
    return prompts


def _resolve_config(arguments: argparse.Namespace) -> Any:
    """Load the run config, preferring the copy beside the checkpoint."""
    from avgen.config import load_config

    if arguments.config:
        return load_config(arguments.config)
    beside = Path(arguments.checkpoint)
    candidate = (beside if beside.is_dir() else beside.parent) / "config.yaml"
    if candidate.is_file():
        emit(f"  using the config recorded beside the checkpoint: {candidate}")
        return load_config(candidate)
    emit(
        "  no config found beside the checkpoint; falling back to schema "
        "defaults. Pass --config to reproduce the run's own settings."
    )
    return load_config(None)


def run(arguments: argparse.Namespace) -> int:
    """Generate media from a checkpoint.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    import secrets

    configuration = _resolve_config(arguments)
    inference = configuration.inference
    prompts = _read_prompts(arguments)

    seed = arguments.seed
    if seed < 0:
        seed = inference.seed if inference.seed >= 0 else secrets.randbelow(2**31)

    settings = {
        "sampler": arguments.sampler or inference.sampler,
        "steps": arguments.steps or inference.steps,
        "guidance": (
            inference.guidance if arguments.guidance is None else arguments.guidance
        ),
        "guidance_rescale": (
            inference.guidance_rescale
            if arguments.guidance_rescale is None
            else arguments.guidance_rescale
        ),
        "negative_prompt": (
            inference.negative_prompt
            if arguments.negative_prompt is None
            else arguments.negative_prompt
        ),
        "seed": seed,
        "frames": arguments.frames or inference.frames,
        "height": arguments.height or inference.height,
        "width": arguments.width or inference.width,
        "batch_size": arguments.batch_size or inference.batch_size,
    }
    output = Path(arguments.output or inference.output)

    emit(rule("avgen generate", character="="))
    emit(f"  checkpoint    {arguments.checkpoint}")
    emit(f"  prompts       {len(prompts)}")
    emit(
        f"  sampling      {settings['sampler']} x{settings['steps']} "
        f"cfg={settings['guidance']} rescale={settings['guidance_rescale']} "
        f"seed={settings['seed']}"
    )
    emit(
        f"  geometry      {settings['frames']}x{settings['height']}x"
        f"{settings['width']} latent · batch {settings['batch_size']}"
    )
    emit(f"  output        {output}")
    emit(
        "  record these settings with the samples: two clips generated at "
        "different steps or"
    )
    emit("  guidance are not a comparison of anything.")

    if arguments.print_settings:
        return 0

    from avgen.cli._wiring import require_subsystem

    pipeline_class = require_subsystem("avgen.infer.pipeline", "GenerationPipeline")
    from_checkpoint = getattr(pipeline_class, "from_checkpoint", None)
    if from_checkpoint is None:
        return fail(
            "avgen.infer.GenerationPipeline has no from_checkpoint constructor; "
            "CONTRACTS.md §4 declares the pipeline but not how it is loaded "
            "from a checkpoint. Build it yourself and call it directly until "
            "that contract is added."
        )

    pipeline = from_checkpoint(arguments.checkpoint, config=configuration)
    media = pipeline(
        prompts,
        steps=settings["steps"],
        guidance=settings["guidance"],
        seed=settings["seed"],
        negative_prompt=settings["negative_prompt"],
        frames=settings["frames"],
        height=settings["height"],
        width=settings["width"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    written = media.save(output) if hasattr(media, "save") else output
    emit()
    emit(f"  wrote {written}")
    return 0
