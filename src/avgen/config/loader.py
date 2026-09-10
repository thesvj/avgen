"""YAML loading, ``_base_`` composition, env interpolation, dotted overrides.

The whole config system is this file plus :mod:`avgen.config.schema`. It
deliberately reimplements the four features people install a config framework
for, because each one is twenty lines and none of them is worth a dependency
the reader has to learn:

**Composition.** ``_base_: ../model/dit_2b.yaml`` deep-merges a parent file
under this one. Lists replace wholesale rather than merging element-wise —
merging a list of buckets index-by-index is a footgun, because inserting one
bucket at the front silently rewrites every override that followed.

**Environment interpolation.** ``${env:AVGEN_DATA_ROOT}`` and
``${env:WANDB_PROJECT:avgen}`` (with a default). No arbitrary expression
language: a config file that can execute is a config file you have to audit.

**Dotted overrides.** ``train.lr=1e-4``, ``parallel.context=8``,
``data.buckets[0].height=512``. Type coercion is driven by the *dataclass field
annotation*, never by guessing at the string. This matters more than it sounds:
YAML 1.1 parses ``1e-4`` as the string ``"1e-4"`` because it lacks the decimal
point, so a guessing loader silently hands a string to an optimizer. Reading
the annotation makes it a float because the field says ``float``.

**Unknown keys are fatal.** A silently ignored typo in a config is a wasted
cluster run: ``lr_warmup_steps`` instead of ``warmup_steps`` produces a job that
trains happily with the wrong schedule and nothing anywhere says so. Every
unknown key raises, and the message includes the closest valid field name.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
import re
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

import yaml

from avgen.config.schema import RunConfig

__all__ = [
    "ConfigError",
    "apply_overrides",
    "load_config",
    "load_mapping",
    "parse_override",
]

T = TypeVar("T")

#: Key naming one or more parent files to compose under this one.
BASE_KEY = "_base_"

#: ``${env:NAME}`` or ``${env:NAME:default}``. The default may contain anything
#: but a closing brace, which is enough for paths, URLs, and numbers.
_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

#: ``name`` or ``name[3]`` in a dotted override path.
_SEGMENT_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")

_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


class ConfigError(ValueError):
    """A configuration file, override, or field value that avgen refuses.

    A distinct type so the CLI can print it as one clean line rather than a
    traceback: config errors are user errors, and a traceback trains people to
    stop reading error messages.
    """


# ---------------------------------------------------------------------------
# YAML reading, composition, interpolation
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a mapping, with file-level error context."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level; got "
            f"{type(payload).__name__}"
        )
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested mappings.

    Lists are replaced, not merged. Element-wise list merging looks helpful
    until someone prepends a bucket and every downstream index shifts by one,
    at which point the config means something nobody intended and nothing
    warns.

    Args:
        base: The parent mapping.
        override: The child mapping, which wins on conflict.

    Returns:
        A new merged mapping; neither input is modified.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _interpolate(value: Any, *, where: str) -> Any:
    """Substitute ``${env:...}`` references throughout a loaded structure."""
    if isinstance(value, str):
        return _interpolate_string(value, where=where)
    if isinstance(value, dict):
        return {
            key: _interpolate(item, where=f"{where}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _interpolate(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _interpolate_string(text: str, *, where: str) -> str:
    """Substitute env references in one string, failing loudly on a missing var."""

    def _replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(
            f"{where} references environment variable {name!r}, which is not "
            f"set and has no default; write ${{env:{name}:some-default}} if an "
            "absent variable is acceptable"
        )

    return _ENV_PATTERN.sub(_replace, text)


def load_mapping(path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file, resolve ``_base_`` composition, and interpolate.

    Args:
        path: Config file.
        _seen: Files already on the composition stack, used to detect cycles.

    Returns:
        The composed, interpolated mapping. ``_base_`` is stripped.

    Raises:
        ConfigError: If the file is missing, malformed, composes cyclically, or
            references an unset environment variable.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")
    resolved = resolved.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, resolved))
        raise ConfigError(f"_base_ composition is cyclic: {chain}")

    payload = _read_yaml(resolved)
    bases = payload.pop(BASE_KEY, None)
    merged: dict[str, Any] = {}
    if bases is not None:
        entries = [bases] if isinstance(bases, str) else bases
        if not isinstance(entries, list):
            raise ConfigError(
                f"{resolved}: {BASE_KEY} must be a path or a list of paths; got "
                f"{bases!r}"
            )
        for entry in entries:
            if not isinstance(entry, str):
                raise ConfigError(
                    f"{resolved}: every {BASE_KEY} entry must be a string path; "
                    f"got {entry!r}"
                )
            # Relative to the file that names it, so a config tree can be moved
            # or vendored without rewriting every path inside it.
            base_path = (resolved.parent / entry).resolve()
            merged = deep_merge(
                merged, load_mapping(base_path, _seen=(*_seen, resolved))
            )
    merged = deep_merge(merged, payload)
    return typing.cast(dict[str, Any], _interpolate(merged, where=str(resolved.name)))


# ---------------------------------------------------------------------------
# Dotted overrides
# ---------------------------------------------------------------------------


def parse_override(text: str) -> tuple[list[str | int], Any]:
    """Split one ``key.path=value`` override into a path and a raw value.

    The value is parsed with ``yaml.safe_load`` so lists (``[a, b]``), mappings
    and nulls work, but a bare scalar that YAML would leave as a string stays a
    string — final typing is the annotation's job in :func:`coerce`.

    Args:
        text: The override, e.g. ``"data.buckets[0].height=512"``.

    Returns:
        The parsed path segments (strings for fields, ints for list indices)
        and the raw value.

    Raises:
        ConfigError: If the override has no ``=``, an empty key, or a malformed
            path segment.
    """
    if "=" not in text:
        raise ConfigError(
            f"override {text!r} is not of the form key.path=value; "
            "for example train.lr=1e-4 or parallel.context=8"
        )
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r} has an empty key")

    segments: list[str | int] = []
    for part in key.split("."):
        match = _SEGMENT_PATTERN.match(part)
        if match is None:
            raise ConfigError(
                f"override {text!r} has a malformed path segment {part!r}; "
                "segments look like 'name' or 'name[0]'"
            )
        segments.append(match.group(1))
        for index in re.findall(r"\[(\d+)\]", match.group(2)):
            segments.append(int(index))

    stripped = raw.strip()
    if stripped in ("null", "~", "None"):
        return segments, None
    if stripped == "":
        return segments, ""
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        # A value YAML cannot parse is simply a string; the annotation decides
        # what it becomes.
        value = raw
    if value is None:
        value = raw
    return segments, value


def apply_overrides(
    mapping: dict[str, Any],
    overrides: Sequence[str],
) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides to a raw config mapping.

    Applied to the *mapping*, before dataclass construction, so an override is
    indistinguishable from having written the value in the file — including
    being subject to unknown-key rejection and to ``__post_init__`` validation.

    Args:
        mapping: The composed mapping.
        overrides: Override strings, applied left to right.

    Returns:
        A new mapping with the overrides applied.

    Raises:
        ConfigError: If an override is malformed, indexes a non-list, or
            indexes past the end of a list.
    """
    result = dict(mapping)
    for text in overrides:
        segments, value = parse_override(text)
        cursor: Any = result
        for position, segment in enumerate(segments[:-1]):
            cursor = _descend(cursor, segment, text, segments[:position])
        _assign(cursor, segments[-1], value, text)
    return result


def _path_string(segments: Sequence[str | int]) -> str:
    """Render a parsed override path back into its dotted form."""
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, int):
            parts[-1] = f"{parts[-1]}[{segment}]" if parts else f"[{segment}]"
        else:
            parts.append(segment)
    return ".".join(parts)


def _descend(
    cursor: Any,
    segment: str | int,
    text: str,
    so_far: Sequence[str | int],
) -> Any:
    """Walk one level into the mapping, creating absent sections as needed."""
    where = _path_string(so_far) or "<root>"
    if isinstance(segment, int):
        if cursor == {}:
            # The section was absent from the file, so _descend synthesised an
            # empty mapping for it. An index into a list that does not exist
            # cannot be created from the command line: the list's other entries
            # would have to be invented.
            raise ConfigError(
                f"override {text!r} indexes {where}[{segment}] but {where} is "
                "not present in the config file; define the list in YAML first "
                "— an entry conjured from the command line would have no "
                "siblings and no defaults you chose"
            )
        if not isinstance(cursor, list):
            raise ConfigError(
                f"override {text!r} indexes {where} with [{segment}] but that "
                f"key holds {type(cursor).__name__}, not a list"
            )
        if segment >= len(cursor):
            raise ConfigError(
                f"override {text!r} indexes {where}[{segment}] but only "
                f"{len(cursor)} entries exist; add the entry to the YAML file "
                "rather than creating it from the command line"
            )
        return cursor[segment]
    if not isinstance(cursor, dict):
        raise ConfigError(
            f"override {text!r} descends into {where} but that path holds "
            f"{type(cursor).__name__}, not a mapping"
        )
    # A section absent from the file is still a valid override target: its
    # fields have defaults. An invalid *field* is caught by the unknown-key
    # check once the dataclass is built.
    if segment not in cursor:
        cursor[segment] = {}
    return cursor[segment]


def _assign(cursor: Any, segment: str | int, value: Any, text: str) -> None:
    """Write the final segment of an override path."""
    if isinstance(segment, int):
        if not isinstance(cursor, list):
            raise ConfigError(
                f"override {text!r} assigns to index [{segment}] of a "
                f"{type(cursor).__name__}, not a list"
            )
        if segment >= len(cursor):
            raise ConfigError(
                f"override {text!r} assigns to index [{segment}] but only "
                f"{len(cursor)} entries exist"
            )
        cursor[segment] = value
        return
    if not isinstance(cursor, dict):
        raise ConfigError(
            f"override {text!r} assigns {segment!r} on a "
            f"{type(cursor).__name__}, not a mapping"
        )
    cursor[segment] = value


# ---------------------------------------------------------------------------
# Annotation-driven construction
# ---------------------------------------------------------------------------


def _hints(cls: type) -> dict[str, Any]:
    """Resolve a dataclass's annotations, which ``from __future__`` stringifies."""
    return typing.get_type_hints(cls)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return the non-``None`` member of an optional annotation, and a flag."""
    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0], True
        # A genuine multi-type union has no single coercion target; leave the
        # value alone and let __post_init__ reject it with a field-level message.
        return Any, True
    return annotation, False


def _coerce_bool(where: str, value: Any) -> bool:
    """Coerce a YAML scalar or CLI string to a bool, or refuse clearly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _TRUE:
        return True
    if isinstance(value, str) and value.lower() in _FALSE:
        return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ConfigError(
        f"{where} must be a boolean (true/false); got {value!r}"
    )


def _coerce_number(where: str, value: Any, target: type) -> Any:
    """Coerce to int or float, rejecting silent truncation."""
    if isinstance(value, bool):
        raise ConfigError(f"{where} must be a {target.__name__}; got {value!r}")
    if target is float:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as error:
                raise ConfigError(
                    f"{where} must be a float; got {value!r}"
                ) from error
    if target is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # Truncating 3.7 to 3 for a "steps" field is exactly the sort of
            # silent reinterpretation that makes two runs incomparable.
            if value.is_integer():
                return int(value)
            raise ConfigError(
                f"{where} must be an integer; got {value!r}, which would have "
                "to be truncated"
            )
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError as error:
                    raise ConfigError(
                        f"{where} must be an integer; got {value!r}"
                    ) from error
                if parsed.is_integer():
                    return int(parsed)
                raise ConfigError(
                    f"{where} must be an integer; got {value!r}"
                ) from None
    raise ConfigError(f"{where} must be a {target.__name__}; got {value!r}")


def coerce(where: str, value: Any, annotation: Any) -> Any:
    """Convert a raw YAML/CLI value to what a dataclass field is annotated as.

    Args:
        where: Dotted field path, echoed in every error message.
        value: The raw value from YAML or from an override.
        annotation: The resolved field annotation.

    Returns:
        The converted value.

    Raises:
        ConfigError: If the value cannot be converted, or is a mapping with
            keys the target dataclass does not define.
    """
    annotation, optional = _unwrap_optional(annotation)
    if value is None:
        if optional or annotation is Any:
            return None
        raise ConfigError(f"{where} may not be null")
    if annotation is Any:
        return value

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        if not isinstance(value, Mapping):
            raise ConfigError(
                f"{where} must be a mapping of "
                f"{annotation.__name__} fields; got {type(value).__name__}"
            )
        return build(annotation, value, where=where)

    origin = get_origin(annotation)
    if origin is Literal:
        options = get_args(annotation)
        if value not in options:
            raise ConfigError(
                f"{where} must be one of {', '.join(map(repr, options))}; "
                f"got {value!r}"
            )
        return value
    if origin in (tuple, list):
        args = get_args(annotation)
        if not isinstance(value, list | tuple):
            raise ConfigError(
                f"{where} must be a list; got {type(value).__name__}"
            )
        item_type: Any = args[0] if args else Any
        items = [
            coerce(f"{where}[{index}]", item, item_type)
            for index, item in enumerate(value)
        ]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(
                f"{where} must be a mapping; got {type(value).__name__}"
            )
        args = get_args(annotation)
        value_type: Any = args[1] if len(args) == 2 else Any
        return {
            str(key): coerce(f"{where}.{key}", item, value_type)
            for key, item in value.items()
        }

    if annotation is bool:
        return _coerce_bool(where, value)
    if annotation in (int, float):
        return _coerce_number(where, value, annotation)
    if annotation is str:
        if isinstance(value, str):
            return value
        if isinstance(value, int | float) and not isinstance(value, bool):
            # A path or name that YAML read as a number, e.g. run_name: 2024.
            return str(value)
        raise ConfigError(f"{where} must be a string; got {value!r}")
    return value


def _suggest(key: str, valid: Sequence[str]) -> str:
    """Return a ``did you mean`` clause for an unknown key, when one is close."""
    close = difflib.get_close_matches(key, valid, n=1, cutoff=0.6)
    return f"; did you mean {close[0]!r}?" if close else ""


def build(cls: type[T], mapping: Mapping[str, Any], *, where: str = "") -> T:
    """Construct a config dataclass from a mapping, rejecting unknown keys.

    Args:
        cls: The dataclass to build.
        mapping: Field values.
        where: Dotted prefix for error messages.

    Returns:
        The constructed instance.

    Raises:
        ConfigError: If a key is unknown, a value cannot be coerced, or the
            dataclass's own validation rejects the result.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = _hints(cls)
    names = [f.name for f in dataclasses.fields(cls) if f.init]
    prefix = f"{where}." if where else ""

    unknown = [key for key in mapping if key not in names]
    if unknown:
        # A silently ignored key is a wasted cluster run. Name every one, and
        # point at the nearest real field so the fix is a single edit.
        details = ", ".join(
            f"{prefix}{key}{_suggest(key, names)}" for key in sorted(unknown)
        )
        raise ConfigError(
            f"unknown configuration key(s) in {where or cls.__name__}: {details}. "
            f"Valid keys: {', '.join(names)}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in mapping.items():
        kwargs[key] = coerce(f"{prefix}{key}", value, hints[key])
    try:
        return typing.cast(T, cls(**kwargs))
    except ConfigError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        location = f" in {where}" if where else ""
        raise ConfigError(f"invalid configuration{location}: {error}") from error


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Sequence[str] = (),
) -> RunConfig:
    """Load, compose, override, and validate a run configuration.

    Args:
        path: YAML file. ``None`` starts from the schema defaults, which is how
            ``avgen plan`` runs with no config file at all.
        overrides: Dotted ``key=value`` strings applied after composition, e.g.
            ``("train.lr=1e-4", "parallel.context=8")``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing or malformed, an override is
            malformed, a key is unknown, or a field fails validation.
    """
    mapping = load_mapping(path) if path is not None else {}
    if overrides:
        mapping = apply_overrides(mapping, overrides)
    return build(RunConfig, mapping)
