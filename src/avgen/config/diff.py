"""Serialising a config, and comparing two of them field by field.

Both halves exist for the same reason: **a run whose exact configuration is not
recorded next to its checkpoint is not reproducible**, and a checkpoint that is
not reproducible is a checkpoint you cannot publish, bisect, or resume with
confidence. So :func:`save_config` writes the *resolved* config — after
``_base_`` composition, after environment interpolation, after every
command-line override — beside every checkpoint. The file on disk is the whole
truth, with nothing left to reconstruct from shell history.

:func:`config_diff` is the other half. "Run B is better than run A" is only a
claim if you can say what differed, and the honest answer is usually longer than
the one people remember. The diff is field-precise and dotted, so it pastes
straight into a commit message or an experiment log.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from avgen.config.schema import RunConfig

__all__ = ["config_diff", "format_diff", "save_config", "to_mapping"]


def to_mapping(config: Any) -> Any:
    """Convert a config dataclass tree into plain JSON/YAML-safe containers.

    ``dataclasses.asdict`` would nearly do, but it leaves tuples as tuples,
    which ``yaml.safe_dump`` refuses. Converting to lists here means the written
    file round-trips back through :func:`avgen.config.load_config` unchanged,
    which is asserted in the test suite — a config that cannot be reloaded is
    a config that does not document anything.

    Args:
        config: A config dataclass, or any nested container of them.

    Returns:
        Dicts, lists, and scalars only.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return {
            field.name: to_mapping(getattr(config, field.name))
            for field in dataclasses.fields(config)
            if field.init
        }
    if isinstance(config, Mapping):
        return {str(key): to_mapping(value) for key, value in config.items()}
    if isinstance(config, tuple | list):
        return [to_mapping(item) for item in config]
    return config


def save_config(config: RunConfig, path: str | Path) -> Path:
    """Write a resolved configuration to a YAML file.

    Args:
        config: The configuration to record.
        path: Destination file. Parent directories are created.

    Returns:
        The written path.

    Raises:
        OSError: If the file cannot be written. Deliberately not swallowed:
            failing to record the config of a run that is about to consume a
            thousand GPU-hours is worth stopping for.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# avgen resolved run configuration.\n"
        "# Written after _base_ composition, environment interpolation, and\n"
        "# every command-line override. Load it directly to reproduce the run:\n"
        f"#   avgen train --config {destination.name}\n"
    )
    body = yaml.safe_dump(
        to_mapping(config),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=88,
    )
    destination.write_text(header + body, encoding="utf-8")
    return destination


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted paths, with list indices inline."""
    flat: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
        return flat
    flat[prefix] = value
    return flat


def config_diff(a: RunConfig, b: RunConfig) -> dict[str, tuple[Any, Any]]:
    """Return every field where two configurations differ.

    Args:
        a: The reference configuration.
        b: The candidate configuration.

    Returns:
        Dotted field path to ``(value_in_a, value_in_b)``. A field present in
        only one side appears with the sentinel string ``"<absent>"`` on the
        other, which happens when the two configs have different-length lists
        such as data buckets.
    """
    left = _flatten(to_mapping(a))
    right = _flatten(to_mapping(b))
    absent = "<absent>"
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(left) | set(right)):
        first = left.get(key, absent)
        second = right.get(key, absent)
        if first != second:
            differences[key] = (first, second)
    return differences


def format_diff(
    differences: Mapping[str, tuple[Any, Any]],
    *,
    left_name: str = "a",
    right_name: str = "b",
) -> str:
    """Render a diff as an aligned, pasteable block.

    Args:
        differences: Output of :func:`config_diff`.
        left_name: Label for the first configuration.
        right_name: Label for the second.

    Returns:
        A multi-line string, or a single line stating the configs are identical.
    """
    if not differences:
        return f"{left_name} and {right_name} are identical"
    width = max(len(key) for key in differences)
    lines = [f"{'field'.ljust(width)}  {left_name}  ->  {right_name}"]
    lines.append("-" * (width + len(left_name) + len(right_name) + 10))
    for key, (first, second) in differences.items():
        lines.append(f"{key.ljust(width)}  {first!r}  ->  {second!r}")
    return "\n".join(lines)
