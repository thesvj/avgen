## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Why

<!--
What was broken, missing, or wrong. If this is a performance change, the number
goes here. If you rejected an alternative approach, say which and why it lost —
that is the part a reviewer cannot reconstruct.
-->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance
- [ ] Documentation
- [ ] Build / CI
- [ ] **Breaking change** — describe the migration below

Related issue:

## How this was verified

- [ ] `make lint`
- [ ] `make type`
- [ ] `make test`
- [ ] `make docs` — the build is `--strict`, so a docstring documenting a
      parameter that does not exist fails it
- [ ] New tests cover the change (a bug fix includes the test that fails without it)
- [ ] If you changed a public signature, the `examples/` that use it still run
      (`pytest tests/test_examples.py`) — they are the first code a new user copies
- [ ] Ran on a GPU — describe how many, and which:
- [ ] Ran multi-node — describe world size and topology:

## Parallelism impact

<!--
Fill this in if you touched src/avgen/parallel/**, a model's
tensor_parallel_plan, a model's model_shape, or anything that changes memory or
communication. Otherwise write "none".

Run `make simulate` and paste the before/after table. A reviewer cannot
otherwise tell whether predicted MFU at 1024 ranks moved by 1% or by 30%.
-->

```
paste `make simulate` output here, before and after
```

- [ ] `make simulate` passes
- [ ] Mesh dimension order `(pp, dp_replicate, dp_shard, cp, tp)` is unchanged,
      or the change is justified above
- [ ] Checkpoints written before this change still load after it, at a
      different rank count

## Scale-rule checklist

<!-- CONTRACTS.md §6. These pass every test you can run locally and fail on a cluster. -->

- [ ] RNG varies on `data_rank` only
- [ ] Data shards on `data_rank` only
- [ ] Losses and metrics reduce over the `dp_cp` mesh, not the whole world
- [ ] No `.item()` inside a training step
- [ ] Gradient clipping goes through `avgen.parallel.clip_grad_norm`
- [ ] Optional dependencies are imported lazily, with a `RuntimeError` naming
      the extra
- [ ] No `torch.cuda` call at import time
- [ ] Same seed + config + rank count still reproduces the same loss curve

## Housekeeping

- [ ] `CHANGELOG.md` updated under `## [Unreleased]`, or this is user-invisible
- [ ] Docs under `docs/` updated in this PR, or this changes no behaviour
- [ ] `CONTRACTS.md` updated if a cross-subsystem contract changed
- [ ] Public symbols have Google-style docstrings, and comments explain *why*
