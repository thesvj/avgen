"""Shared CLI plumbing: exit codes, error rendering, formatting, table layout.

Two conventions the whole CLI keeps.

**Errors are one line, not a traceback.** A stack trace is for a bug in avgen. A
missing file, an unknown config key, a world size that does not factor — those
are the user's, and printing forty lines of frames for them teaches people to
stop reading errors. ``--traceback`` restores the frames when the error really
is a bug.

**Exit codes mean something**, so a shell script can branch on them:

===  =========================================================
0    Success.
1    A runtime error — the operation was understood and failed.
2    A usage error — argparse rejected the arguments.
3    A configuration error — the file or an override is invalid.
130  Interrupted (SIGINT). The conventional 128 + SIGINT.
===  =========================================================
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

__all__ = [
    "EXIT_CONFIG",
    "EXIT_ERROR",
    "EXIT_INTERRUPT",
    "EXIT_OK",
    "EXIT_USAGE",
    "add_config_arguments",
    "add_traceback_argument",
    "emit",
    "fail",
    "human_bytes",
    "human_count",
    "human_seconds",
    "load_run_config",
    "rule",
    "table",
]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_INTERRUPT = 130


def emit(text: str = "") -> None:
    """Write a line to stdout.

    The house style bans ``print`` in the library because logging belongs to
    telemetry. A command-line tool's output *is* its return value, so the CLI
    keeps exactly one place where writing to stdout happens and everything else
    goes through it.

    Args:
        text: The line.
    """
    sys.stdout.write(f"{text}\n")


def fail(message: str, *, code: int = EXIT_ERROR) -> int:
    """Print one clean error line to stderr and return an exit code.

    Args:
        message: The error, already phrased for a human.
        code: Exit code to return.

    Returns:
        ``code``, so a command can ``return fail(...)``.
    """
    sys.stderr.write(f"avgen: error: {message}\n")
    return code


def add_traceback_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--traceback`` flag to a parser.

    Args:
        parser: The parser or subparser.
    """
    parser.add_argument(
        "--traceback",
        action="store_true",
        help=(
            "print the full Python traceback instead of a single error line. "
            "Use it when you suspect a bug in avgen rather than a mistake in "
            "your invocation."
        ),
    )


def add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
) -> None:
    """Add ``--config`` and the trailing dotted-override arguments.

    Args:
        parser: The subcommand parser.
        required: Whether a config file must be supplied.
    """
    parser.add_argument(
        "--config",
        metavar="PATH",
        required=required,
        help=(
            "YAML run configuration. Compose one from the shipped pieces with "
            "'_base_:' — see configs/README.md."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "dotted overrides applied after the file, e.g. train.lr=1e-4 "
            "parallel.context=8 data.buckets[0].height=512. Types come from "
            "the schema, so 1e-4 becomes a float because the field is a float. "
            "An unknown key is an error, never a silent no-op."
        ),
    )


def load_run_config(arguments: argparse.Namespace) -> Any:
    """Load the configuration named by ``--config`` plus any overrides.

    Args:
        arguments: Parsed arguments carrying ``config`` and ``overrides``.

    Returns:
        The validated :class:`avgen.config.RunConfig`.

    Raises:
        avgen.config.ConfigError: If the file or an override is invalid.
    """
    from avgen.config import load_config

    return load_config(
        getattr(arguments, "config", None),
        overrides=tuple(getattr(arguments, "overrides", ()) or ()),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def human_count(value: float) -> str:
    """Format a large count with a K/M/B/T suffix.

    Args:
        value: The number.

    Returns:
        A short string, e.g. ``"2.05 B"``.
    """
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.2f} {suffix}"
    return f"{value:.0f}"


def human_bytes(value: float) -> str:
    """Format a byte count in binary units.

    Args:
        value: Bytes.

    Returns:
        A short string, e.g. ``"63.4 GiB"``.
    """
    for threshold, suffix in (
        (1024.0**4, "TiB"),
        (1024.0**3, "GiB"),
        (1024.0**2, "MiB"),
        (1024.0, "KiB"),
    ):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f} {suffix}"
    return f"{value:.0f} B"


def human_seconds(value: float) -> str:
    """Format a duration with a unit that keeps three significant figures.

    Args:
        value: Seconds.

    Returns:
        A short string, e.g. ``"412 ms"`` or ``"1.83 s"``.
    """
    if value >= 1.0:
        return f"{value:.2f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.1f} ms"
    return f"{value * 1e6:.0f} us"


def rule(title: str = "", width: int = 78, character: str = "-") -> str:
    """Return a horizontal rule, optionally with an inline title.

    Args:
        title: Text embedded at the left of the rule.
        width: Total width.
        character: Fill character.

    Returns:
        The rule.
    """
    if not title:
        return character * width
    prefix = f"{character * 2} {title} "
    return prefix + character * max(0, width - len(prefix))


def table(
    columns: Sequence[tuple[str, str]],
    rows: Sequence[dict[str, Any]],
    *,
    aligns: Sequence[str] = (),
) -> str:
    """Render rows as a fixed-width table sized to its content.

    Sizing to content rather than to hard-coded widths matters more than it
    looks: a plan table with a truncated ``bottleneck`` column hides the single
    most useful field in the output.

    Args:
        columns: ``(key, header)`` pairs, in display order.
        rows: Mappings keyed by the column keys.
        aligns: Per-column alignment, ``"l"`` or ``"r"``. Defaults to left for
            the first column and right for the rest, which is what makes a
            column of numbers readable.

    Returns:
        A multi-line string with a header and a separator.
    """
    if not rows:
        return "(no rows)"
    alignment = list(aligns) or ["l"] + ["r"] * (len(columns) - 1)
    widths = [
        max(len(header), *(len(str(row.get(key, ""))) for row in rows))
        for key, header in columns
    ]

    def render(values: Sequence[str]) -> str:
        cells = [
            value.rjust(width) if align == "r" else value.ljust(width)
            for value, width, align in zip(values, widths, alignment, strict=True)
        ]
        return "  ".join(cells).rstrip()

    lines = [render([header for _, header in columns])]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(render([str(row.get(key, "")) for key, _ in columns]))
    return "\n".join(lines)
