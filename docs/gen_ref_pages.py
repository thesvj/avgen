"""Generate the API reference from the source tree at documentation build time.

Hand-written reference stubs rot. Someone adds a module, nobody adds the page,
and the site quietly stops describing the package. This script walks
``src/avgen`` on every build, writes one mkdocstrings stub per module, and
writes the ``SUMMARY.md`` that ``mkdocs-literate-nav`` turns into the reference
navigation tree. Nothing under ``docs/reference/`` is committed — it is
generated and gitignored.

Run automatically by the ``mkdocs-gen-files`` plugin; see ``mkdocs.yml``.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

#: Package root relative to the repository root.
SOURCE_ROOT = Path("src")

#: Where generated pages land inside ``docs/``.
REFERENCE_ROOT = Path("reference")

#: Short blurbs for the top-level subpackages, so the reference index is
#: navigable without opening every page. Keys are the subpackage name.
SUBPACKAGE_SUMMARY: dict[str, str] = {
    "core": "Token streams, batches, patchification, RNG, state, metrics.",
    "parallel": "The five parallelism axes, the device mesh, and the plans.",
    "simulate": "Price and validate a parallelism plan without a cluster.",
    "models": "Diffusion transformers and the model registry.",
    "train": "Objective, timestep sampling, optimizer, schedule, trainer.",
    "data": "Resumable sources, buckets, shards, and the loader.",
    "checkpoint": "Distributed Checkpoint save/load and export.",
    "telemetry": "Loggers, throughput, memory reporting, profiling.",
    "infer": "Samplers, guidance, and the generation pipeline.",
    "codecs": "Video and audio codecs and text encoders.",
    "finetune": "LoRA, DoRA, freezing, and control adapters.",
    "rl": "Flow-GRPO, DPO, and reward models.",
    "eval": "Metrics and evaluation reports.",
    "config": "Dataclass and YAML configuration.",
    "cli": "The ``avgen`` command line.",
}


def module_identifier(path: Path) -> tuple[str, ...]:
    """Return the dotted module parts for a source file.

    Args:
        path: Path to a ``.py`` file, relative to :data:`SOURCE_ROOT`.

    Returns:
        The module path as a tuple of identifiers, with ``__init__`` collapsed
        into its package.
    """
    parts = tuple(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def should_skip(parts: tuple[str, ...]) -> bool:
    """Whether a module is internal and should not appear in the reference.

    Modules whose name begins with an underscore are internal by the house
    convention in ``CONTRACTS.md`` §2, and ``__main__`` is an entry point
    rather than an API.

    Args:
        parts: Dotted module parts.

    Returns:
        Whether to skip the module.
    """
    if not parts:
        return True
    if parts[-1] == "__main__":
        return True
    return any(part.startswith("_") for part in parts)


def write_reference_pages() -> list[tuple[str, ...]]:
    """Write one mkdocstrings stub per public module.

    Returns:
        The dotted parts of every module that got a page, sorted.
    """
    written: list[tuple[str, ...]] = []
    for source in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = source.relative_to(SOURCE_ROOT)
        parts = module_identifier(relative)
        if should_skip(parts):
            continue

        # Drop the leading "avgen" so URLs read /reference/core/tokens/ rather
        # than /reference/avgen/core/tokens/. The top-level package itself is
        # rendered on the reference overview page instead of getting its own.
        if len(parts) == 1:
            continue
        tail = Path(*parts[1:])
        doc_path = (
            tail / "index.md"
            if relative.name == "__init__.py"
            else tail.with_suffix(".md")
        )
        full_doc_path = REFERENCE_ROOT / doc_path

        identifier = ".".join(parts)
        with mkdocs_gen_files.open(full_doc_path, "w") as handle:
            print(f"# `{identifier}`", file=handle)
            print("", file=handle)
            print(f"::: {identifier}", file=handle)

        # Makes the "edit this page" link point at the source, which is the
        # only thing anyone would actually want to edit here.
        mkdocs_gen_files.set_edit_path(full_doc_path, Path("..") / source)
        written.append(parts)
    return sorted(written)


def write_index(modules: list[tuple[str, ...]]) -> None:
    """Write the reference landing page.

    Args:
        modules: Every module that got a page.
    """
    subpackages = sorted({parts[1] for parts in modules if len(parts) > 1})
    with mkdocs_gen_files.open(REFERENCE_ROOT / "index.md", "w") as handle:
        print("# API reference", file=handle)
        print("", file=handle)
        print(
            "Generated from the source tree on every documentation build. "
            "Modules whose name starts with an underscore are internal and "
            "are not listed.",
            file=handle,
        )
        print("", file=handle)
        print("| Subpackage | What lives there |", file=handle)
        print("| --- | --- |", file=handle)
        for name in subpackages:
            summary = SUBPACKAGE_SUMMARY.get(name, "")
            print(f"| [`avgen.{name}`]({name}/index.md) | {summary} |", file=handle)
        print("", file=handle)
        print("## `avgen`", file=handle)
        print("", file=handle)
        print("::: avgen", file=handle)


def write_summary(modules: list[tuple[str, ...]]) -> None:
    """Write the ``SUMMARY.md`` consumed by ``mkdocs-literate-nav``.

    Args:
        modules: Every module that got a page.
    """
    with mkdocs_gen_files.open(REFERENCE_ROOT / "SUMMARY.md", "w") as handle:
        print("- [Overview](index.md)", file=handle)
        current: str | None = None
        for parts in modules:
            if len(parts) == 1:
                continue
            package = parts[1]
            if package != current:
                current = package
                print(f"- avgen.{package}", file=handle)
                print(f"    - [Overview]({package}/index.md)", file=handle)
            if len(parts) > 2:
                leaf = "/".join(parts[2:])
                print(f"    - [{parts[-1]}]({package}/{leaf}.md)", file=handle)


modules = write_reference_pages()
write_index(modules)
write_summary(modules)
