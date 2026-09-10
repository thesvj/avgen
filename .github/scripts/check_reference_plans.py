#!/usr/bin/env python3
"""Check avgen's reference parallelism plans without touching a cluster.

Why this exists
---------------

A parallelism plan for a 1024-GPU job has to be right *before* anyone spends
1024 GPU-hours discovering that it is not. The failure modes are exactly the
ones that do not appear at small scale: an activation policy that leaves the
model four gigabytes over budget, a mesh whose tensor-parallel dimension
straddles two nodes, a change to a model's shape that quietly doubles the
context-parallel ring volume.

Nothing in a CPU unit test notices any of that. This script does, in seconds,
by doing three separate things for every reference plan:

1. **Search.** Enumerate every valid factorisation of the world size, price
   each one with the memory, communication, and compute models, and confirm
   that at least one still fits. "Nothing fits any more" is the loudest
   possible regression signal and the easiest one to miss.
2. **Price the pinned plan.** Each reference entry pins the degrees a real job
   would launch with. Check that it fits in device memory and that its
   predicted MFU and scaling efficiency have not fallen through the floors in
   ``reference_plans.json``.
3. **Build the mesh for real.** Under ``FakeProcessGroup``, construct the
   actual ``DeviceMesh`` at the full world size and assert the dimension order
   is ``(pp, dp_replicate, dp_shard, cp, tp)``. That ordering is a performance
   decision — it is what keeps tensor-parallel traffic inside an NVLink domain
   — and getting it wrong produces no error at runtime, just a slower job.

The floors are a tripwire, not a target. They are set loose on purpose and
tightened by a maintainer once a real run has calibrated the cost models.

Usage::

    make simulate
    python .github/scripts/check_reference_plans.py --world-sizes 8 64 512 1024
    python .github/scripts/check_reference_plans.py --update   # print observed values
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "reference_plans.json"

#: Mesh dimensions in the order avgen builds them. Duplicated here on purpose:
#: if someone reorders MESH_DIM_ORDER in the library, this check must fail
#: rather than agree with the change.
EXPECTED_MESH_ORDER: tuple[str, ...] = ("pp", "dp_replicate", "dp_shard", "cp", "tp")


class PlanCheckError(Exception):
    """Raised when a reference plan violates one of its recorded floors."""


def load_config(path: Path) -> dict[str, Any]:
    """Read the reference-plan definition file.

    Args:
        path: Path to ``reference_plans.json``.

    Returns:
        The parsed document.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"reference plan file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        document: dict[str, Any] = json.load(handle)
    return document


def build_shape(entry: dict[str, Any]) -> Any:
    """Turn a JSON model entry into a :class:`ModelShape`.

    Args:
        entry: One value from the ``models`` object.

    Returns:
        The model shape.
    """
    from avgen.simulate.memory import ModelShape

    return ModelShape(
        parameters=int(entry["parameters"]),
        depth=int(entry["depth"]),
        width=int(entry["width"]),
        sequence_length=int(entry["sequence_length"]),
        micro_batch_size=int(entry.get("micro_batch_size", 1)),
        mlp_ratio=int(entry.get("mlp_ratio", 4)),
        num_heads=int(entry.get("num_heads", 16)),
        text_tokens=int(entry.get("text_tokens", 0)),
    )


def resolve_accelerator(name: str) -> Any:
    """Look up a device profile by name.

    Args:
        name: One of ``A100_80GB``, ``H100_SXM``, ``H200_SXM``, ``B200``.

    Returns:
        The accelerator profile.

    Raises:
        KeyError: If the name is unknown.
    """
    from avgen.simulate import compute

    profiles = {
        "A100_80GB": compute.A100_80GB,
        "H100_SXM": compute.H100_SXM,
        "H200_SXM": compute.H200_SXM,
        "B200": compute.B200,
    }
    if name not in profiles:
        raise KeyError(f"unknown accelerator {name!r}; known: {sorted(profiles)}")
    return profiles[name]


def check_mesh_order(dims: Any, *, device_type: str | None) -> tuple[str, ...]:
    """Build the real mesh under a fake process group and validate its order.

    Args:
        dims: The parallelism degrees.
        device_type: Mesh device type, or ``None`` to auto-detect.

    Returns:
        The mesh dimension names actually produced.

    Raises:
        PlanCheckError: If the ordering does not match ``EXPECTED_MESH_ORDER``.
    """
    from avgen.simulate.world import fake_world

    with fake_world(dims, rank=0, device_type=device_type) as world:
        mesh = world.require_mesh()
        names = tuple(mesh.mesh_dim_names or ())

    expected = tuple(name for name in EXPECTED_MESH_ORDER if name in names)
    if names != expected:
        raise PlanCheckError(
            f"mesh dimension order is {names}, expected {expected}. "
            "Tensor parallelism must be the innermost dimension so its ranks "
            "share an NVLink domain; pipeline parallelism must be outermost "
            "because its traffic is a small point-to-point handoff."
        )
    return names


def evaluate_plan(
    plan: dict[str, Any],
    models: dict[str, Any],
    defaults: dict[str, Any],
    *,
    device_type: str | None,
) -> dict[str, Any]:
    """Search, price, and structurally validate one reference plan.

    Args:
        plan: One entry from the ``plans`` array.
        models: The ``models`` object, for shape lookup.
        defaults: The ``defaults`` object.
        device_type: Mesh device type, or ``None`` to auto-detect.

    Returns:
        A JSON-safe row describing what was observed.

    Raises:
        PlanCheckError: If any floor is violated or nothing fits.
    """
    from avgen.parallel.dims import ParallelDims
    from avgen.simulate.comms import (
        INFINIBAND_NDR,
        NVLINK4,
        estimate_step_communication,
    )
    from avgen.simulate.compute import estimate_compute
    from avgen.simulate.memory import estimate_memory
    from avgen.simulate.plan import SearchSpace, search_parallel_plan

    name = str(plan["name"])
    shape = build_shape(models[plan["model"]])
    accelerator = resolve_accelerator(
        str(plan.get("accelerator", defaults.get("accelerator", "H100_SXM")))
    )
    gpus_per_node = int(plan.get("gpus_per_node", defaults.get("gpus_per_node", 8)))
    headroom = float(defaults.get("memory_headroom", 0.10))
    achieved = float(defaults.get("achieved_fraction", 0.45))
    floors = dict(plan.get("floors", {}))
    world_size = int(plan["world_size"])

    # 1. Does anything at all still fit at this world size?
    space = SearchSpace(
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        max_tensor=int(plan.get("max_tensor", gpus_per_node)),
        max_context=int(plan.get("max_context", 16)),
        max_pipeline=int(plan.get("max_pipeline", 1)),
    )
    candidates = search_parallel_plan(
        shape,
        space,
        accelerator=accelerator,
        achieved_fraction=achieved,
        headroom=headroom,
        top_k=5,
    )
    if not candidates:
        raise PlanCheckError(
            f"[{name}] no parallelism configuration fits at world_size="
            f"{world_size}. Either the memory model changed, the model shape "
            "grew, or the search bounds are now too tight."
        )
    best = candidates[0]

    # 2. Price the degrees a real job would actually launch with.
    pinned_spec = dict(plan.get("pinned", {}))
    dims = ParallelDims(
        world_size=world_size,
        dp_replicate=int(pinned_spec.get("dp_replicate", 1)),
        dp_shard=int(pinned_spec.get("dp_shard", -1)),
        tensor=int(pinned_spec.get("tensor", 1)),
        context=int(pinned_spec.get("context", 1)),
        pipeline=int(pinned_spec.get("pipeline", 1)),
    )
    pinned_memory = estimate_memory(shape, dims)
    pinned_compute = estimate_compute(
        shape, dims, accelerator=accelerator, achieved_fraction=achieved
    )
    pinned_comms = estimate_step_communication(
        shape,
        dims,
        intra_node=NVLINK4,
        inter_node=INFINIBAND_NDR,
        compute_seconds=pinned_compute.seconds,
    )
    step_seconds = pinned_comms.estimated_step_seconds
    pinned_mfu = pinned_compute.mfu(step_seconds, accelerator.peak_flops())

    # 3. The mesh a launcher would actually build.
    mesh_names = check_mesh_order(dims, device_type=device_type)

    row = {
        "name": name,
        "world_size": world_size,
        "sequence_length": shape.sequence_length,
        "pinned_plan": dims.describe(),
        "mesh_dims": "/".join(mesh_names) or "single-device",
        "pinned_memory_gib": round(pinned_memory.total_gib, 2),
        "pinned_dominant_term": pinned_memory.dominant_term(),
        "pinned_step_seconds": round(step_seconds, 4),
        "pinned_mfu": round(pinned_mfu, 4),
        "pinned_scaling_efficiency": round(pinned_comms.scaling_efficiency, 4),
        "pinned_bottleneck": pinned_comms.bottleneck(),
        "best_search_plan": best.dims.describe(),
        "best_search_mfu": round(best.mfu(accelerator), 4),
        "candidates_that_fit": len(candidates),
    }

    failures: list[str] = []
    max_memory = float(floors.get("max_memory_gib", accelerator.memory_gib))
    if not pinned_memory.fits_in(accelerator.memory_gib, headroom=headroom):
        failures.append(
            f"pinned plan needs {pinned_memory.total_gib:.1f} GiB, which does "
            f"not fit in {accelerator.memory_gib:.0f} GiB with "
            f"{headroom:.0%} headroom; dominant term is "
            f"{pinned_memory.dominant_term()}"
        )
    if pinned_memory.total_gib > max_memory:
        failures.append(
            f"pinned memory {pinned_memory.total_gib:.1f} GiB exceeds the "
            f"recorded ceiling of {max_memory:.1f} GiB"
        )
    if "min_mfu" in floors and pinned_mfu < float(floors["min_mfu"]):
        failures.append(
            f"predicted MFU regressed to {pinned_mfu:.3f}, below the floor of "
            f"{float(floors['min_mfu']):.3f}"
        )
    min_eff = floors.get("min_scaling_efficiency")
    if min_eff is not None and pinned_comms.scaling_efficiency < float(min_eff):
        failures.append(
            f"scaling efficiency regressed to "
            f"{pinned_comms.scaling_efficiency:.3f}, below the floor of "
            f"{float(min_eff):.3f}; the dominant collective is "
            f"{pinned_comms.bottleneck()}"
        )

    row["failures"] = failures
    return row


def render_table(rows: list[dict[str, Any]]) -> str:
    """Render the observed rows as a Markdown table.

    Args:
        rows: Rows from :func:`evaluate_plan`.

    Returns:
        A Markdown table.
    """
    columns = [
        ("name", "plan"),
        ("world_size", "world"),
        ("pinned_plan", "degrees"),
        ("pinned_memory_gib", "mem GiB"),
        ("pinned_step_seconds", "step s"),
        ("pinned_mfu", "MFU"),
        ("pinned_scaling_efficiency", "scaling"),
        ("pinned_bottleneck", "bottleneck"),
        ("candidates_that_fit", "fits"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    return "\n".join(lines)


def write_step_summary(text: str) -> None:
    """Append text to the GitHub Actions step summary when running in CI.

    Args:
        text: Markdown to append.
    """
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, or ``None`` to use ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Reference plan definition file.",
    )
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Only check plans at these world sizes. Default: all of them.",
    )
    parser.add_argument(
        "--device-type",
        default=None,
        help="Mesh device type. Default: cuda when a GPU is present, else cpu.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Print observed values as JSON so a maintainer can refresh floors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check every reference plan and report.

    Args:
        argv: Argument vector, or ``None`` to use ``sys.argv``.

    Returns:
        Process exit code: 0 when every plan holds, 1 otherwise.
    """
    args = parse_args(argv)
    document = load_config(args.config)
    models = dict(document["models"])
    defaults = dict(document.get("defaults", {}))
    plans = [
        plan
        for plan in document["plans"]
        if args.world_sizes is None or int(plan["world_size"]) in args.world_sizes
    ]
    if not plans:
        print(f"no reference plans matched world sizes {args.world_sizes}")
        return 1

    rows: list[dict[str, Any]] = []
    hard_errors: list[str] = []
    for plan in plans:
        try:
            rows.append(
                evaluate_plan(
                    plan, models, defaults, device_type=args.device_type
                )
            )
        except PlanCheckError as exc:
            hard_errors.append(str(exc))

    if rows:
        table = render_table(rows)
        print(table)
        write_step_summary("## Reference parallelism plans\n\n" + table)

    if args.update:
        print("\nObserved values (paste into reference_plans.json floors):")
        print(json.dumps(rows, indent=2, sort_keys=True))

    failures = [
        f"[{row['name']}] {message}"
        for row in rows
        for message in row["failures"]
    ]
    failures.extend(hard_errors)

    if failures:
        print("\nFAILED: a reference parallelism plan regressed.\n")
        for message in failures:
            print(f"  - {message}")
        print(
            "\nIf this regression is intended, update the floors in "
            f"{args.config.relative_to(REPO_ROOT)} in the same pull request "
            "and say why in the description. Do not silence it separately."
        )
        write_step_summary(
            "\n### Failures\n\n"
            + "\n".join(f"- {message}" for message in failures)
        )
        return 1

    print(f"\nOK: {len(rows)} reference plan(s) still fit and hold their floors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
