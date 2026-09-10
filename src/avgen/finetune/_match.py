"""Name matching shared by adapter injection and freezing.

Every fine-tuning entry point in this package selects a subset of a model by
*name*: which linears get a LoRA, which parameters stay trainable, which blocks
the control tower copies. One matching rule, defined once here, keeps a recipe
portable — a pattern written for ``freeze_except`` selects the same modules when
handed to ``LoRAConfig.target_modules``.

The rule is deliberately more permissive than a bare glob, because the two
things users actually write are very different in shape::

    "q_proj"                    # a bare submodule name, meaning "every q_proj"
    "blocks.3?.*"               # a glob over the full dotted path

Supporting only globs would force the first form to be written
``"*.q_proj"``, which is easy to get wrong (it silently fails to match a
top-level module) and reads badly in a config file. Supporting only substrings
would make ``"norm"`` match ``attention_norm``, ``ffn_norm`` *and*
``normalisation_stats`` with no way to be precise.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence

__all__ = ["matches_any", "normalize_patterns"]

#: Characters that make a pattern a glob rather than a literal name.
_GLOB_CHARACTERS = frozenset("*?[]")


def normalize_patterns(patterns: Iterable[str]) -> tuple[str, ...]:
    """Validate and freeze a pattern collection into a deterministic tuple.

    Order is preserved and duplicates are dropped in first-seen order rather
    than via a ``set``, because pattern order is observable (the first match
    wins in reporting) and set iteration order is not reproducible across
    interpreter runs — which rule 12 of the build contract forbids for anything
    that affects computation.

    Args:
        patterns: Patterns to normalise.

    Returns:
        The patterns, de-duplicated, in first-seen order.

    Raises:
        TypeError: If ``patterns`` is a bare string. Passing ``"q_proj"`` where
            a sequence is expected would iterate its characters and silently
            match almost everything, so it is rejected loudly.
        ValueError: If the collection is empty or contains an empty pattern.
    """
    if isinstance(patterns, str):
        raise TypeError(
            "patterns must be a sequence of strings, not a single string; "
            f"pass ('{patterns}',) rather than '{patterns}'"
        )
    seen: dict[str, None] = {}
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string; got {pattern!r}")
        if not pattern:
            raise ValueError("pattern must be non-empty")
        seen[pattern] = None
    if not seen:
        raise ValueError("at least one pattern is required")
    return tuple(seen)


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    """Test a dotted module or parameter name against a pattern collection.

    A pattern matches when either:

    * it is a glob (contains ``*``, ``?`` or ``[]``) that matches the whole
      dotted name, case-sensitively; or
    * it is a literal that equals one of the dot-separated components of the
      name, or equals a dot-separated *suffix* of it.

    The suffix rule is what makes ``"attention.q_proj"`` select
    ``blocks.7.attention.q_proj`` without a leading wildcard, and the component
    rule is what makes ``"q_proj"`` select every query projection in the model
    while leaving ``q_proj_scale`` alone.

    Args:
        name: Dotted name, as produced by ``named_modules`` or
            ``named_parameters``.
        patterns: Patterns to test against.

    Returns:
        Whether any pattern matches.
    """
    components = name.split(".")
    for pattern in patterns:
        if _GLOB_CHARACTERS & set(pattern):
            if fnmatch.fnmatchcase(name, pattern):
                return True
            continue
        if pattern in components:
            return True
        parts = pattern.split(".")
        if len(parts) <= len(components) and components[-len(parts) :] == parts:
            return True
    return False
