# Adding a parallelism plan

A parallelism plan is a function from `(model, ParallelDims, DeviceMesh)` to a
transformed model. `parallelize()` composes the built-in ones; this page is
about writing a new one, or changing an existing one.

**This is the highest-risk change in the codebase.** A plan bug does not crash.
It produces a job that runs, trains, and is slower or subtly wrong, at a scale
where nobody can bisect it interactively. Everything below exists to catch that
before it reaches a cluster.

## The application order is fixed

```python
parallelize(model, dims, mesh=mesh, config=config)
```

applies, in this order:

1. **Tensor parallel** — needs the unsharded module structure to address
   submodules by path.
2. **Activation checkpointing** — wraps blocks; must see the TP-transformed
   module so it checkpoints the right thing.
3. **`torch.compile`** — per block, before FSDP, so it compiles compute rather
   than communication.
4. **FSDP2 `fully_shard`** — last, so it shards what everything else produced.

Reordering breaks things quietly. FSDP before TP shards parameters that TP then
tries to shard again. Compiling after FSDP compiles the all-gather into the
graph. If your new plan has an ordering requirement, say so in a comment naming
what breaks without it.

## Writing one

```python
def apply_my_thing(model: nn.Module, dims: ParallelDims, mesh: DeviceMesh) -> nn.Module:
    """Apply my transformation.

    Args:
        model: The model to transform, possibly already transformed by an
            earlier stage.
        dims: Validated parallelism degrees.
        mesh: The device mesh, indexed by dimension name.

    Returns:
        The transformed model.
    """
    if not dims.my_axis_enabled:
        return model  # a degenerate degree is a no-op
    submesh = mesh["my_axis"]
    ...
    return model
```

Four requirements:

**Index the mesh by name, never by position.** `mesh["cp"]`, not `mesh[2]`. Only
dimensions greater than one appear in the mesh, so positions shift depending on
the plan.

**Be a no-op when the degree is one.** That is what lets the same plan code run
unchanged on a single GPU, and it is what the enablement properties
(`dims.cp_enabled`, `dims.tp_enabled`, …) exist for. Compare against those, not
against `1`.

**Be idempotent under a re-apply**, or raise clearly. Re-application happens in
tests and in resume paths.

**Assume no GPU.** The plan must apply under `FakeProcessGroup` on a CPU-only
machine, because that is how it gets tested at 1024 ranks. No `torch.cuda` call
at import time, and none in the plan unless it is guarded.

## Mesh ordering is not yours to change

```text
(pp, dp_replicate, dp_shard, cp, tp)
```

Rank ordering makes the last dimension vary fastest, so `tp` ranks are adjacent
and land inside one NVLink domain, and `pp` ranks are furthest apart — correct,
because pipeline traffic is a small point-to-point handoff while TP traffic is
an all-reduce on every layer.

Changing this order needs a discussion first. It is asserted in CI by
`.github/scripts/check_reference_plans.py`, and the assertion exists because the
failure mode is a job that runs at half speed with no error message.

## Flattened meshes

Two views are registered because they are needed constantly:

- **`dp_shard_cp`** — FSDP2 shards parameters across both the sharded-data and
  context dimensions. CP ranks hold different tokens of the same sample, so they
  can hold different parameter shards. Use this for anything sharding
  parameters.
- **`dp_cp`** — the ranks holding different data or different sequence shards.
  Use this for reducing losses and metrics. Never reduce over the whole world:
  TP and PP ranks hold the *same* loss, so the result would be wrong by a
  constant factor.

```python
from avgen.parallel import all_reduce_mean, data_mesh

loss = all_reduce_mean(loss, data_mesh(mesh))
```

## Test it in the simulator, at scale

This is the part that is not optional.

```python
from avgen.parallel import ParallelDims
from avgen.simulate.world import simulate_rank


def test_my_plan_at_1024():
    dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
    report = simulate_rank(
        dims,
        build=lambda: MyDiT(config),  # built on meta; free
        apply_plan=lambda m, d, mesh: apply_my_thing(m, d, mesh),
    )
    assert report["shard_efficiency"] > 0.9
```

Check at least: 8, 64, 512, and 1024 ranks, and at least one plan where `tp > 1`
and one where `pp > 1`. Bugs concentrate at the boundaries — the first world
size where a dimension crosses a node, and the first where a degenerate
dimension disappears from the mesh.

Also check the collective *pattern*, not just that it applies. `CommDebugMode`
counts and attributes collectives by module; one giant all-gather where there
should be 32 per-block ones applies cleanly and destroys overlap.

## Then price it

```bash
make simulate
```

runs `.github/scripts/check_reference_plans.py` at all four reference world
sizes: it searches the space, prices the pinned plan, checks memory and MFU
against recorded floors, and asserts the mesh order. **Paste the before/after
table into your pull request** — a reviewer cannot otherwise tell whether you
moved predicted MFU at 1024 ranks by 1% or by 30%.

CI runs the identical script, so a green local run means a green CI run.

If your change intentionally moves a number past a floor, update
`.github/reference_plans.json` in the same pull request and say why. Do not
silence it in a separate change.

## The checklist

- [ ] No-op when the degree is one; compares against `dims.*_enabled`
- [ ] Indexes the mesh by name
- [ ] Idempotent, or raises clearly on re-application
- [ ] Applies under `FakeProcessGroup` with no GPU
- [ ] Uses `dp_shard_cp` for parameters, `dp_cp` for loss reduction
- [ ] Ordering requirements documented in a comment saying what breaks
- [ ] Simulator tests at 8 / 64 / 512 / 1024, with `tp > 1` and `pp > 1` covered
- [ ] Collective counts checked, not just that it applies
- [ ] `make simulate` green; before/after table in the PR
- [ ] Loss curve verified unchanged against the previous plan on a small config

That last one deserves emphasis. Run the same config with the old plan and the
new one at small scale and compare the loss curves. They should match within
noise. **A plan that changes the loss is a plan that changes the mathematics**,
and that is either a bug or a change that needs to be declared as one.

## Further reading

- [Parallelism](../guides/parallelism.md)
- [Simulation](../guides/simulation.md)
- [Contracts](contracts.md)
