"""``avgen data {synthesize,shard,inspect,validate}`` — the dataset side of the CLI.

``synthesize``
    Write a deterministic synthetic dataset with no optional dependency at all.
    Its purpose is not to train a useful model; it is to make every other
    command runnable end-to-end on a laptop in seconds, so a contributor can
    verify a change without a dataset, a decoder, or a GPU.

``shard``
    Pack encoded latents into mmap + safetensors shards with a sha256 manifest
    and an atomic commit. The atomicity is the part that matters: a shard writer
    interrupted mid-file must leave no half-written shard that a later run reads
    as valid, because the corruption then shows up as a mysterious loss spike a
    week later.

``inspect``
    Report what a shard directory contains — sample count, bucket distribution,
    codec fingerprints, token totals — without reading the payloads.

``validate``
    Verify the manifest against the files and check that every sample's shape
    agrees with its declared bucket. **Run this before a long job.** A shard set
    that is 0.1% corrupt trains fine and produces a model that is slightly worse
    for no visible reason.

**This module ships templates, not data.** No dataset names, sources, quotas, or
tuned thresholds appear anywhere in avgen; the shipped data configs are blanks
for you to fill in.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avgen.cli._common import (
    add_config_arguments,
    add_traceback_argument,
    emit,
    fail,
    human_count,
    load_run_config,
    rule,
    table,
)

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``data`` subcommand group.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "data",
        help="synthesize, shard, inspect, or validate training data",
        description=(
            "Dataset tooling. avgen trains on pre-encoded latent shards; the "
            "encode step is offline and belongs to whatever pipeline produces "
            "your data.\n\n"
            "avgen ships no dataset. The configs under configs/data/ are "
            "templates with the paths left blank."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(dest="action", metavar="ACTION")

    synthesize = actions.add_parser(
        "synthesize",
        help="write a deterministic synthetic dataset (no dependencies)",
        description=(
            "Generate a small synthetic latent dataset so every other command "
            "is runnable end to end without real data, a decoder, or a GPU.\n\n"
            "It is deterministic given the seed, so a smoke test is a real "
            "regression test rather than a coin flip."
        ),
    )
    add_config_arguments(synthesize, required=False)
    synthesize.add_argument(
        "--output", metavar="PATH", required=True, help="destination directory"
    )
    synthesize.add_argument(
        "--samples",
        type=int,
        default=64,
        help="samples to write (default: %(default)s)",
    )
    synthesize.add_argument("--seed", type=int, default=0, help="generator seed")
    add_traceback_argument(synthesize)
    synthesize.set_defaults(handler=run, action="synthesize")

    shard = actions.add_parser(
        "shard",
        help="pack encoded latents into mmap + safetensors shards",
        description=(
            "Write shards with a sha256 manifest and an atomic COMMIT. A writer "
            "interrupted mid-shard leaves nothing a later run will read as "
            "valid — silent partial shards surface as an unexplained loss "
            "regression days later."
        ),
    )
    shard.add_argument("input", metavar="SRC", help="directory of encoded latents")
    shard.add_argument("output", metavar="DST", help="shard directory to write")
    shard.add_argument(
        "--samples-per-shard",
        type=int,
        default=1024,
        help=(
            "samples per shard file. Larger shards read faster and resume more "
            "coarsely (default: %(default)s)"
        ),
    )
    add_traceback_argument(shard)
    shard.set_defaults(handler=run, action="shard")

    inspect = actions.add_parser(
        "inspect",
        help="report a shard directory's contents without reading payloads",
    )
    inspect.add_argument("path", metavar="PATH", help="shard directory")
    add_traceback_argument(inspect)
    inspect.set_defaults(handler=run, action="inspect")

    validate = actions.add_parser(
        "validate",
        help="verify manifest hashes and per-sample shapes",
        description=(
            "Check every shard against the manifest and every sample against "
            "its declared bucket. Run this before a long job: a shard set that "
            "is a fraction of a percent corrupt trains without complaint and "
            "produces a model that is slightly worse for no visible reason."
        ),
    )
    validate.add_argument("path", metavar="PATH", help="shard directory")
    validate.add_argument(
        "--config",
        metavar="PATH",
        default="",
        help="run configuration whose data.buckets the shapes are checked against",
    )
    validate.add_argument(
        "--sample",
        type=int,
        default=0,
        help=(
            "check only this many samples. 0 checks everything, which is what "
            "you want before a long run (default: %(default)s)"
        ),
    )
    add_traceback_argument(validate)
    validate.set_defaults(handler=run, action="validate")

    parser.set_defaults(handler=run)
    return parser


def run(arguments: argparse.Namespace) -> int:
    """Dispatch to the requested data action.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    action = getattr(arguments, "action", None)
    if action is None:
        return fail(
            "avgen data needs an action: synthesize, shard, inspect, or validate"
        )

    from avgen.cli._wiring import require_subsystem

    if action == "synthesize":
        configuration = load_run_config(arguments)
        source_class = require_subsystem("avgen.data.synthetic", "SyntheticSource")
        writer = require_subsystem("avgen.data.shard", "write_shard")
        destination = Path(arguments.output)
        destination.mkdir(parents=True, exist_ok=True)
        source = source_class(configuration.data, seed=arguments.seed)
        written = writer(destination, source, limit=arguments.samples)
        emit(rule("avgen data synthesize", character="="))
        emit(f"  wrote {arguments.samples} samples to {destination}")
        emit(f"  shards: {written}")
        emit()
        emit("  point a config at it with:")
        emit("      avgen train --config configs/train/smoke_cpu.yaml \\")
        emit(f"          data.source=latent_shards data.root={destination}")
        return 0

    if action == "shard":
        writer = require_subsystem("avgen.data.shard", "write_shard")
        written = writer(
            Path(arguments.output),
            Path(arguments.input),
            samples_per_shard=arguments.samples_per_shard,
        )
        emit(f"  wrote {written} to {arguments.output}")
        return 0

    path = Path(arguments.path)
    if not path.exists():
        return fail(f"shard directory not found: {path}")

    if action == "inspect":
        return _inspect(path)
    if action == "validate":
        return _validate(path, arguments)
    return fail(f"unknown data action {action!r}")


def _inspect(path: Path) -> int:
    """Print a shard directory's contents."""
    from avgen.cli._wiring import require_subsystem

    read_shard = require_subsystem("avgen.data.shard", "read_shard")
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return fail(
            f"{path} contains no .safetensors shards. If this is a directory of "
            "raw media, encode it first — avgen trains on latents, not pixels."
        )

    rows: list[dict[str, Any]] = []
    total = 0
    for shard in shards:
        info = read_shard(shard, metadata_only=True)
        count = int(info.get("samples", 0))
        total += count
        rows.append(
            {
                "shard": shard.name,
                "samples": str(count),
                "bucket": str(info.get("bucket_id", "?")),
                "video": str(info.get("video_shape", "?")),
                "audio": str(info.get("audio_shape", "?")),
                "codec": str(info.get("video_codec_id", "?"))[:24],
            }
        )
    emit(rule(f"shards in {path}", character="="))
    emit(
        table(
            [
                ("shard", "shard"),
                ("samples", "samples"),
                ("bucket", "bucket"),
                ("video", "video shape"),
                ("audio", "audio shape"),
                ("codec", "video codec"),
            ],
            rows,
        )
    )
    emit()
    emit(f"  {len(shards)} shard(s) · {human_count(total)} samples")
    codecs = {row["codec"] for row in rows}
    if len(codecs) > 1:
        emit(
            "  WARNING: more than one video codec fingerprint is present. "
            "Latents from two"
        )
        emit(
            "  different autoencoders are not interchangeable, and mixing them "
            "trains a model"
        )
        emit("  that is worse than either half alone.")
        return 1
    return 0


def _validate(path: Path, arguments: argparse.Namespace) -> int:
    """Verify manifest hashes and per-sample shapes."""
    from avgen.cli._wiring import require_subsystem

    validate_shards = require_subsystem("avgen.data.shard", "validate_shards")
    buckets = None
    if arguments.config:
        buckets = load_run_config(arguments).data.buckets

    result = validate_shards(path, buckets=buckets, limit=arguments.sample or None)
    failures = list(result.get("failures", ()))
    emit(rule(f"validating {path}", character="="))
    emit(f"  shards checked   {result.get('shards', 0)}")
    emit(f"  samples checked  {human_count(result.get('samples', 0))}")
    emit(f"  failures         {len(failures)}")
    for failure in failures[:20]:
        emit(f"    - {failure}")
    if len(failures) > 20:
        emit(f"    ... and {len(failures) - 20} more")
    if failures:
        emit()
        emit(
            "  Do not train on this. A shard set that is a fraction of a "
            "percent corrupt"
        )
        emit(
            "  trains without complaint and produces a model that is slightly "
            "worse for no"
        )
        emit("  visible reason.")
        return 1
    emit("  all checks passed")
    return 0
