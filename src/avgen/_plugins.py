"""Discovery of third-party extensions advertised through entry points.

Every registry in avgen — models, metrics, samplers, rewards — exists so that
the common extension is a *pure addition*: install a package, name the thing in
a config, and it works without a fork and without an import in your training
script. Entry points are what make that true.

Two properties are load-bearing, and both are easy to get wrong:

**Loading is deterministic.** Entry points are sorted by name before loading.
Unordered iteration would let two ranks with identical installed packages
register in a different order, and any registry whose iteration order reaches a
computation is then a silent divergence between ranks.

**A failure is loud.** A plugin that silently fails to load is indistinguishable
from a typo in the name — hours later, on a cluster, in a message that mentions
neither. So a broken entry point raises, naming the group, the name and the
value it could not resolve.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from importlib import metadata
from typing import Any, TypeVar

__all__ = ["load_entry_points"]

_T = TypeVar("_T")

#: Groups already loaded. A registry's loader is called from every lookup, so
#: this is what keeps it O(1) after the first.
_LOADED: set[str] = set()


def load_entry_points(
    group: str,
    registry: MutableMapping[str, _T],
    *,
    kind: str,
    validate: Callable[[Any], bool] | None = None,
    expected: str = "",
) -> None:
    """Register every plugin advertised in ``group`` that is not already known.

    Args:
        group: Entry-point group, e.g. ``"avgen.metrics"``.
        registry: Mapping the plugin is registered into, keyed by name.
        kind: Human word for what is being loaded, used in error messages
            (``"metric"``, ``"sampler"``, ...).
        validate: Optional predicate on the loaded object. Use it to reject a
            plugin that resolves to the wrong sort of thing at *load* time
            rather than at first use, which is a much later and stranger error.
        expected: Description of what ``validate`` accepts, for the message.

    Raises:
        RuntimeError: If an advertised entry point cannot be imported, or fails
            ``validate``. Both name the group, the entry-point name and its
            value, because the package that advertised it is the thing to fix
            and nothing else in the traceback will say which one it was.
    """
    if group in _LOADED:
        return
    _LOADED.add(group)

    # Sorted for determinism across ranks; see the module docstring.
    for point in sorted(metadata.entry_points(group=group), key=lambda ep: ep.name):
        # An entry point whose object registers itself on import (the usual
        # case, via the @register_* decorator) is already present by the time
        # load() returns, and must not be overwritten by its own re-registration.
        if point.name in registry:
            continue
        try:
            loaded = point.load()
        except Exception as error:
            raise RuntimeError(
                f"failed to load {kind} entry point {point.name!r} "
                f"({point.value}) from group {group!r}"
            ) from error
        if point.name in registry:
            continue
        if validate is not None and not validate(loaded):
            raise RuntimeError(
                f"{kind} entry point {point.name!r} ({point.value}) from group "
                f"{group!r} must resolve to {expected or 'a valid ' + kind}; "
                f"got {loaded!r}"
            )
        registry[point.name] = loaded
