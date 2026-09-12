"""Pick — and price — a parallelism plan for 1024 GPUs, on a laptop.

The most expensive mistake in large-scale training is a plan that does not fit,
or fits and runs at 12% MFU. Finding that out costs an allocation. avgen's
simulator answers it in milliseconds from analytical models of memory,
collectives and compute, with no cluster and no GPU.

Run:
    python examples/02_choose_a_parallelism_plan.py

The same search backs `avgen plan`, and its predictions are calibrated against
real runs — see docs/guides/simulation.md for the accuracy this does and does
not claim.
"""

from __future__ import annotations

from avgen.simulate.compute import H100_SXM
from avgen.simulate.memory import ModelShape
from avgen.simulate.plan import SearchSpace, render_plan_table, search_parallel_plan


def main() -> None:
    # A 5B DiT on ten seconds of 720p at 24fps. The sequence length is what
    # makes video different from language: 120k tokens is where attention stops
    # being a detail and starts being the entire cost model.
    shape = ModelShape(
        parameters=5_000_000_000,
        depth=40,
        width=3072,
        num_heads=24,
        sequence_length=120_000,
        text_tokens=256,
    )
    space = SearchSpace(world_size=1024, gpus_per_node=8, max_context=16, max_tensor=8)

    candidates = search_parallel_plan(shape, space, accelerator=H100_SXM, top_k=5)
    print(render_plan_table(candidates, H100_SXM))

    best = candidates[0]
    print(f"\nbest plan:   {best.dims.describe()}")
    print(f"peak memory: {best.memory.total_gib:.1f} GiB of {H100_SXM.memory_gib} GiB")
    print(f"bottleneck:  {best.communication.bottleneck()}")
    print(
        "\nThe bottleneck field is the useful one: it names the single term that "
        "dominates\nthe step, so you know which knob is worth turning before you "
        "book the nodes."
    )


if __name__ == "__main__":
    main()
