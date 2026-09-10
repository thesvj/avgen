"""``avgen`` — the command-line entry point.

Built on ``argparse`` and nothing else. click and typer are both better
libraries and neither is worth a dependency here: the CLI's job includes being
runnable in a container that has torch and nothing optional, and every
dependency in that path is one more thing that can be the wrong version on a
cluster login node.

**Every subcommand imports its subsystem lazily, inside its handler.** That is
not a micro-optimisation. avgen's subsystems are large and some of them are
still being written; ``avgen --help``, ``avgen info``, and ``avgen plan`` must
work when ``avgen.models`` does not import at all, because those three commands
are how you diagnose that situation. The cost is one import line per handler;
the benefit is a CLI that degrades one command at a time instead of all at once.

Exit codes are in :mod:`avgen.cli._common`. Errors print as a single line unless
``--traceback`` is given, and a keyboard interrupt exits 130 without a stack
trace — a training run stopped on purpose is not a crash.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from avgen.cli import (
    checkpoint as checkpoint_command,
)
from avgen.cli import (
    data as data_command,
)
from avgen.cli import (
    evaluate as evaluate_command,
)
from avgen.cli import (
    generate as generate_command,
)
from avgen.cli import (
    info as info_command,
)
from avgen.cli import (
    plan as plan_command,
)
from avgen.cli import (
    simulate as simulate_command,
)
from avgen.cli import (
    train as train_command,
)
from avgen.cli._common import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_USAGE,
    fail,
)

__all__ = ["build_parser", "main"]

_DESCRIPTION = """\
avgen — train video and audio-video generation models at scale.

Start here:
  avgen plan --world-size 512 --seq-len 65536 --params 2e9 --depth 32 --width 2560
      Rank every parallelism plan for a model shape. No config, no GPU, <1s.

  avgen info
      What is installed, what is importable, what hardware is visible.

  avgen train --config configs/train/single_gpu.yaml train.lr=1e-4
      Train. Any config value is overridable on the command line.

Configuration is plain YAML over plain dataclasses — no hydra, no omegaconf.
Files compose with '_base_:' and are overridden with dotted key=value pairs.
See configs/README.md.
"""

_EPILOG = """\
exit codes:
  0    success
  1    runtime error
  2    usage error
  3    configuration error (bad file, bad override, unknown key)
  130  interrupted

Every command accepts --traceback to print Python frames instead of one line.
"""


def build_parser() -> argparse.ArgumentParser:
    """Assemble the full argument parser.

    Returns:
        The parser, with every subcommand registered and a ``handler`` default
        on each.
    """
    parser = argparse.ArgumentParser(
        prog="avgen",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the avgen version and exit",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="print full Python tracebacks instead of a single error line",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    for module in (
        train_command,
        generate_command,
        evaluate_command,
        simulate_command,
        plan_command,
        checkpoint_command,
        data_command,
        info_command,
    ):
        module.add_parser(subparsers)
    return parser


def _version() -> str:
    """Return the installed avgen version, or a marker when not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("avgen")
    except PackageNotFoundError:
        return "0.0.0+unknown (running from a source tree, not an install)"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the avgen command line.

    Args:
        argv: Arguments excluding the program name. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        A process exit code; see :mod:`avgen.cli._common`.
    """
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(arguments, "version", False):
        sys.stdout.write(f"avgen {_version()}\n")
        return 0

    handler = getattr(arguments, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE

    show_traceback = bool(getattr(arguments, "traceback", False))
    try:
        return int(handler(arguments))
    except KeyboardInterrupt:
        # Not a crash. A person stopped a job on purpose, and printing a
        # traceback for that trains them to ignore tracebacks.
        sys.stderr.write("\navgen: interrupted\n")
        return EXIT_INTERRUPT
    except Exception as error:
        if show_traceback:
            raise
        from avgen.config.loader import ConfigError

        code = EXIT_CONFIG if isinstance(error, ConfigError) else EXIT_ERROR
        detail = str(error) or type(error).__name__
        hint = "" if show_traceback else "  (re-run with --traceback for frames)"
        return fail(f"{detail}{hint}", code=code)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
