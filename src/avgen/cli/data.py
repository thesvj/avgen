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

__all__ = ["add_parser", "run", "shard_directories"]


def _shape(descriptor: Any, fields: tuple[str, ...]) -> str:
    """Render selected descriptor dimensions as a compact ``a x b x c``."""
    if descriptor is None:
        return "?"
    return "x".join(str(getattr(descriptor, name, "?")) for name in fields)


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

    if action == "synthesize":
        return _synthesize(arguments)

    if action == "shard":
        return fail(
            "avgen data shard packs a directory of ALREADY ENCODED latents, and "
            "the encode step is dataset-specific — avgen ships no reader for "
            "your layout and will not guess at one. Build a list of "
            "avgen.data.LatentSample in your own encode script and call "
            "avgen.data.write_shard(path, samples); 'avgen data synthesize' "
            "writes a shard set in exactly that format to copy from."
        )

    path = Path(arguments.path)
    if not path.exists():
        return fail(f"shard directory not found: {path}")

    if action == "inspect":
        return _inspect(path)
    if action == "validate":
        return _validate(path, arguments)
    return fail(f"unknown data action {action!r}")


def _synthesize(arguments: argparse.Namespace) -> int:
    """Write a deterministic synthetic shard set.

    It goes through the same ``write_shard`` path a real encode pipeline uses,
    so the result is a genuine shard set — manifest, sha256, atomic commit —
    rather than a special case the loader has to know about. That is what makes
    a smoke run over it a real test of the data path rather than a test of a
    shortcut.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    from avgen.cli._wiring import build_source, require_subsystem

    configuration = load_run_config(arguments)
    write_shard = require_subsystem("avgen.data.shard", "write_shard")
    build_latent_sample = require_subsystem("avgen.data", "build_latent_sample")

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    source = build_source(configuration)

    samples: list[Any] = []
    for batch in source:
        spec = batch.spec
        for index in range(int(batch.video.shape[0])):
            samples.append(
                build_latent_sample(
                    sample_id=int(batch.sample_ids[index]),
                    video=batch.video[index],
                    video_timebase=(
                        spec.video_timebase_num,
                        spec.video_timebase_den,
                    ),
                    video_codec_id=spec.video_codec_id,
                    audio=batch.audio[index] if spec.has_audio else None,
                    audio_timebase=(
                        spec.audio_timebase_num,
                        spec.audio_timebase_den,
                    ),
                    audio_codec_id=spec.audio_codec_id,
                    text=batch.text[index] if spec.has_text else None,
                    text_mask=batch.text_mask[index] if spec.has_text else None,
                    text_width=spec.text_shape[-1] if spec.has_text else 1,
                )
            )
            if len(samples) >= arguments.samples:
                break
        if len(samples) >= arguments.samples:
            break
    if not samples:
        return fail("the configured data source produced no samples")

    manifest = write_shard(destination / "shard-00000", samples)
    emit(rule("avgen data synthesize", character="="))
    emit(f"  wrote {len(samples)} samples to {destination}")
    emit(
        f"  sha256 {getattr(manifest, 'data_sha256', '?')[:16]}... "
        f"({getattr(manifest, 'data_bytes', 0):,} bytes)"
    )
    emit()
    emit("  point a config at it with:")
    emit("      avgen train --config configs/train/smoke_cpu.yaml \\")
    emit(f"          data.source=latent_shards data.root={destination}")
    return 0


def shard_directories(root: Path) -> list[Path]:
    """Find committed shards under a root.

    A shard is a *directory* — container, manifest, and a COMMIT marker written
    in that order — not a single file. Discovery keys on the marker rather than
    on the container, so a shard whose writer was preempted mid-write is simply
    not found. Globbing for the container instead would surface exactly the
    half-written shards the atomic commit exists to hide.

    Args:
        root: Directory to search, recursively.

    Returns:
        Committed shard directories, sorted.
    """
    from avgen.data.shard import COMMIT_FILENAME

    if (root / COMMIT_FILENAME).is_file():
        return [root]
    return sorted(marker.parent for marker in root.glob(f"**/{COMMIT_FILENAME}"))


def _inspect(path: Path) -> int:
    """Print a shard directory's contents."""
    from avgen.cli._wiring import require_subsystem

    shard_reader = require_subsystem("avgen.data.shard", "ShardReader")
    shards = shard_directories(path)
    if not shards:
        return fail(
            f"{path} contains no committed shards. If this is a directory of "
            "raw media, encode it first — avgen trains on latents, not pixels. "
            "If a writer was interrupted, its uncommitted shard is deliberately "
            "invisible here."
        )

    rows: list[dict[str, Any]] = []
    total = 0
    for shard in shards:
        # validate=False: this command reads metadata only, and hashing every
        # payload would turn an inspection into a full validation pass.
        reader = shard_reader(shard, validate=False)
        try:
            records = reader.manifest.records
            total += len(records)
            first = records[0].descriptor if records else None
            rows.append(
                {
                    "shard": shard.name,
                    "samples": str(len(records)),
                    "video": _shape(first, ("video_frames", "height", "width")),
                    "audio": _shape(first, ("audio_channels", "audio_frames")),
                    "codec": str(getattr(first, "video_codec_id", "?"))[:24],
                    "bytes": human_count(reader.manifest.data_bytes),
                }
            )
        finally:
            reader.close()
    emit(rule(f"shards in {path}", character="="))
    emit(
        table(
            [
                ("shard", "shard"),
                ("samples", "samples"),
                ("video", "video shape"),
                ("audio", "audio shape"),
                ("codec", "video codec"),
                ("bytes", "payload"),
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

    validate_shard = require_subsystem("avgen.data.shard", "validate_shard")
    corruption = require_subsystem("avgen.data.shard", "ShardCorruptionError")
    buckets = None
    if arguments.config:
        buckets = {b.name: b for b in load_run_config(arguments).data.buckets}

    shards = shard_directories(path)
    if not shards:
        return fail(f"{path} contains no committed shards")

    failures: list[str] = []
    samples = 0
    for shard in shards:
        try:
            manifest = validate_shard(shard, check_data=True)
        except corruption as error:
            failures.append(f"{shard.name}: {error}")
            continue
        records = manifest.records
        samples += len(records)
        if buckets is None:
            continue
        # A shape that disagrees with every declared bucket means the loader
        # will pad or crop it silently, which changes the token count and
        # therefore the loss normalisation.
        shapes = {
            (r.descriptor.video_frames, r.descriptor.height, r.descriptor.width)
            for r in records
        }
        allowed = {(b.frames, b.height, b.width) for b in buckets.values()}
        for shape in shapes - allowed:
            failures.append(
                f"{shard.name}: video shape {shape} matches no configured "
                f"bucket {sorted(allowed)}"
            )

    emit(rule(f"validating {path}", character="="))
    emit(f"  shards checked   {len(shards)}")
    emit(f"  samples checked  {human_count(samples)}")
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
