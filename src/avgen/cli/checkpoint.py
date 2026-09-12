"""``avgen checkpoint {inspect,convert,export}`` — read, reshard, and release weights.

Three verbs with genuinely different purposes, which is why they are separate
rather than flags on one command:

``inspect``
    Read a checkpoint's metadata without loading it: step, tokens seen, world
    size and parallelism it was saved under, which components are present, and
    the resolved config beside it. The question this answers most often is "is
    the EMA in here", because samples are drawn from EMA weights and a
    checkpoint without them cannot reproduce a published sample.

``convert``
    Reshard across a different rank count. This is what makes a 512-rank
    checkpoint resumable on 64 ranks, which matters whenever a cluster
    allocation changes shape mid-run. DCP does the work; this is the entry
    point that names it.

``export``
    Produce a release artifact — safetensors or a HuggingFace layout — with the
    training state stripped. An exported checkpoint is for inference and cannot
    be resumed from, which is a feature: it is a tenth the size and has no
    optimizer state to go stale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from avgen.cli._common import add_traceback_argument, emit, fail, human_bytes, rule

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``checkpoint`` subcommand group.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "checkpoint",
        help="inspect, reshard, or export a checkpoint",
        description=(
            "Work with checkpoints without launching a training job.\n\n"
            "avgen checkpoints are torch.distributed.checkpoint (DCP) "
            "directories, which reshard across rank counts. Exported artifacts "
            "are safetensors or a HuggingFace layout and cannot be resumed from."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(dest="action", metavar="ACTION")

    inspect = actions.add_parser(
        "inspect",
        help="print a checkpoint's metadata without loading the weights",
        description=(
            "Report step, samples and tokens seen, the parallelism it was saved "
            "under, which components are present (optimizer, schedule, EMA, "
            "data cursor), and the size on disk.\n\n"
            "The EMA line is the one to check before reproducing a sample: "
            "published diffusion samples are drawn from EMA weights and the raw "
            "training weights are noticeably worse."
        ),
    )
    inspect.add_argument("path", metavar="PATH", help="checkpoint directory or file")
    inspect.add_argument(
        "--json", action="store_true", help="emit machine-readable output"
    )
    add_traceback_argument(inspect)
    inspect.set_defaults(handler=run, action="inspect")

    convert = actions.add_parser(
        "convert",
        help="reshard a checkpoint for a different rank count",
        description=(
            "Rewrite a DCP checkpoint so it loads on a different world size or "
            "a different parallelism factorisation. Needed whenever a cluster "
            "allocation changes shape mid-run."
        ),
    )
    convert.add_argument("path", metavar="SRC", help="source checkpoint")
    convert.add_argument("output", metavar="DST", help="destination directory")
    convert.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="rank count the checkpoint should load on",
    )
    add_traceback_argument(convert)
    convert.set_defaults(handler=run, action="convert")

    export = actions.add_parser(
        "export",
        help="export inference weights (safetensors or HuggingFace layout)",
        description=(
            "Strip the training state and write a release artifact.\n\n"
            "Prefer --ema when the checkpoint has EMA weights: they are what "
            "samples should be drawn from. The result is roughly a tenth the "
            "size of the training checkpoint and cannot be resumed from."
        ),
    )
    export.add_argument("path", metavar="SRC", help="source checkpoint")
    export.add_argument("output", metavar="DST", help="destination file or directory")
    export.add_argument(
        "--format",
        default="safetensors",
        choices=("safetensors", "huggingface"),
        help="output layout (default: %(default)s)",
    )
    export.add_argument(
        "--dtype",
        default="",
        choices=("", "float32", "bfloat16", "float16"),
        help="cast weights on export; empty keeps the training dtype",
    )
    export.add_argument(
        "--ema",
        action="store_true",
        help="export the EMA weights rather than the raw training weights",
    )
    export.add_argument(
        "--config",
        default="",
        help=(
            "run config describing the architecture; defaults to config.yaml "
            "beside the checkpoint"
        ),
    )
    add_traceback_argument(export)
    export.set_defaults(handler=run, action="export")

    parser.set_defaults(handler=run)
    return parser


def _directory_size(path: Path) -> int:
    """Return the total size of a checkpoint directory in bytes."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _describe(path: Path) -> dict[str, Any]:
    """Collect what can be learned about a checkpoint without loading weights.

    Deliberately tolerant of an unfamiliar layout: ``inspect`` is most useful on
    the checkpoint you did not produce, and refusing to describe anything it
    does not fully recognise would defeat the purpose.
    """
    description: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": _directory_size(path) if path.exists() else 0,
    }
    if not path.exists():
        return description

    if path.is_dir():
        entries = sorted(item.name for item in path.iterdir())
        description["entries"] = entries
        description["is_dcp"] = any(name.startswith(".metadata") for name in entries)
        config_file = path / "config.yaml"
        if config_file.is_file():
            description["config"] = str(config_file)

    metadata_file = path / "metadata.json" if path.is_dir() else None
    if metadata_file is not None and metadata_file.is_file():
        try:
            description["metadata"] = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            description["metadata_error"] = str(error)

    try:
        manager = __import__(
            "avgen.checkpoint", fromlist=["inspect_checkpoint"]
        ).inspect_checkpoint
    except (ImportError, AttributeError):
        description["detail"] = (
            "avgen.checkpoint.inspect_checkpoint is unavailable, so only "
            "filesystem-level information is shown. CONTRACTS.md §4 declares "
            "save/load/export but not an inspector; that contract is missing."
        )
    else:
        try:
            description.update(manager(path))
        except Exception as error:
            description["detail"] = f"inspector failed: {type(error).__name__}: {error}"
    return description


def _config_beside(path: Path, override: str) -> Any:
    """Load the run config that describes a checkpoint's architecture.

    Args:
        path: The checkpoint directory or file.
        override: An explicit ``--config`` path, which always wins.

    Returns:
        The loaded config, or ``None`` when there is nothing to load. ``None``
        rather than schema defaults: a wrong architecture loads a checkpoint
        into the wrong tensors and the error it raises is about shapes, not
        about the missing config that caused it.
    """
    from avgen.config import load_config

    if override:
        return load_config(override)
    candidate = (path if path.is_dir() else path.parent) / "config.yaml"
    if candidate.is_file():
        emit(f"  using the config recorded beside the checkpoint: {candidate}")
        return load_config(candidate)
    return None


def run(arguments: argparse.Namespace) -> int:
    """Dispatch to the requested checkpoint action.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    action = getattr(arguments, "action", None)
    if action is None:
        return fail("avgen checkpoint needs an action: inspect, convert, or export")

    path = Path(arguments.path)
    if not path.exists():
        return fail(f"checkpoint not found: {path}")

    if action == "inspect":
        description = _describe(path)
        if arguments.json:
            emit(json.dumps(description, indent=2, sort_keys=True, default=str))
            return 0
        emit(rule(f"checkpoint {path}", character="="))
        emit(f"  size          {human_bytes(description['size_bytes'])}")
        for key in ("is_dcp", "config", "detail"):
            if key in description:
                emit(f"  {key:<13} {description[key]}")
        if "entries" in description:
            emit(f"  entries       {', '.join(description['entries'][:12])}")
        for key, value in sorted(description.get("metadata", {}).items()):
            emit(f"  {key:<13} {value}")
        if "ema" not in str(description).lower():
            emit(
                "  note          no EMA weights detected. Samples in the "
                "literature come from"
            )
            emit(
                "                EMA weights; raw training weights are "
                "noticeably worse."
            )
        return 0

    from avgen.cli._wiring import require_subsystem

    if action == "convert":
        convert = require_subsystem("avgen.checkpoint", "convert")
        destination = convert(path, arguments.output, to="safetensors")
        emit(f"  converted {path} -> {destination}")
        emit(
            "  note: a DCP checkpoint already reshards on load, so a run saved "
            f"at one rank count resumes at another. --world-size "
            f"{arguments.world_size} is recorded for reference; this command "
            "changes the storage format, not the shard count."
        )
        return 0

    if action == "export":
        import torch

        # The exporters take a live nn.Module, not a directory: writing
        # safetensors means gathering every (possibly sharded) parameter, which
        # only a constructed model can describe. So the checkpoint has to be
        # materialised into a model first, which needs the architecture — and
        # the architecture comes from the config recorded beside the weights.
        config = _config_beside(path, getattr(arguments, "config", ""))
        if config is None:
            return fail(
                f"no config.yaml beside {path} and none passed with --config. "
                "Exporting builds the model before writing it, so the "
                "architecture has to come from somewhere."
            )

        build_model = require_subsystem("avgen.models", "build_model")
        model_kwargs = require_subsystem("avgen.config.resolve", "model_kwargs")
        model = build_model(config.model.name, model_kwargs(config))
        model.eval()

        if arguments.ema:
            load_ema = require_subsystem("avgen.checkpoint", "load_ema")
            load_ema(path, model, decay=config.train.ema_decay or 0.9999)
            emit("  loaded EMA weights")
        else:
            load_model = require_subsystem("avgen.checkpoint", "load_model")
            load_model(path, model)
            emit(
                "  loaded the live training weights. Published diffusion "
                "samples come from EMA weights; pass --ema when the checkpoint "
                "has them."
            )

        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtypes.get(arguments.dtype)
        if arguments.format == "safetensors":
            export = require_subsystem("avgen.checkpoint", "export_safetensors")
            written = export(
                arguments.output,
                model,
                dtype=dtype,
                metadata={"source": str(path), "ema": str(arguments.ema)},
            )
        else:
            # export_huggingface writes through DCP's storage writer, which owns
            # the file headers; it takes no metadata mapping.
            export = require_subsystem("avgen.checkpoint", "export_huggingface")
            written = export(arguments.output, model, dtype=dtype)
        emit(f"  exported {arguments.format} to {written or arguments.output}")
        emit(
            "  this artifact is for inference and cannot be resumed from; keep "
            "the training checkpoint."
        )
        return 0

    return fail(f"unknown checkpoint action {action!r}")
