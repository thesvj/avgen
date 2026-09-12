# Contributing

avgen is a teaching codebase as much as a training framework. A patch that works
but leaves the next reader guessing *why* is not finished.

The authoritative documents live at the repository root and are normative:

| Document | What it decides |
|---|---|
| [`CONTRIBUTING.md`](https://github.com/thesvj/avgen/blob/main/CONTRIBUTING.md) | Dev setup, the test subsets, house style, how to extend avgen, what a reviewable PR looks like |
| [`CONTRACTS.md`](https://github.com/thesvj/avgen/blob/main/CONTRACTS.md) | The interface agreement between subsystems. Read it before your first change |
| [`GOVERNANCE.md`](https://github.com/thesvj/avgen/blob/main/GOVERNANCE.md) | Who decides what, and the RFC process for changing a frozen contract |
| [`RELEASING.md`](https://github.com/thesvj/avgen/blob/main/RELEASING.md) | How a release is cut, and what each gate exists to prevent |
| [`SUPPORT.md`](https://github.com/thesvj/avgen/blob/main/SUPPORT.md) | Where to ask, and what this project does not promise |
| [`SECURITY.md`](https://github.com/thesvj/avgen/blob/main/SECURITY.md) | Private disclosure. Never the issue tracker |

## The shortest useful path

```bash
git clone https://github.com/thesvj/avgen
cd avgen
make dev          # every extra, dev + docs groups, pre-commit hooks
make test-fast    # ~90s on 16 cores; the gate before you push
```

No GPU is required for any of it, including reviewing a 1024-GPU parallelism
plan — the simulator runs on a CPU.

## The three things reviewers check first

**1. Does the comment say *why*?** Every magic constant, ordering requirement and
numerical choice carries one to three lines saying what breaks without it. A
comment that restates the line below it will be asked for in review, to be
deleted.

**2. Did you run the simulator?** Any change touching `src/avgen/parallel/**`, a
model's `tensor_parallel_plan`, or its `model_shape` changes behaviour at scales
nobody can test interactively:

```bash
make simulate     # reference plans at world sizes 8, 64, 512, 1024
```

Paste the before/after table into the pull request. A reviewer cannot otherwise
tell whether your change moved predicted MFU at 1024 ranks by 1% or by 30%.

**3. Does it hold at scale?** Twelve rules in `CONTRACTS.md` §6 produce bugs that
pass every test you can run on your desk and fail on a cluster — RNG and data
shard on `data_rank` only, losses reduce over the `dp_cp` mesh, never `.item()`
in a training step, and nine more. They are listed in
[`CONTRIBUTING.md`](https://github.com/thesvj/avgen/blob/main/CONTRIBUTING.md#the-rules-that-only-bite-at-scale).

## Sign your commits

avgen uses the [Developer Certificate of Origin](https://developercertificate.org/).
There is no CLA.

```bash
git commit -s -m "parallel: keep tp innermost when pp is enabled"
```

This is a CI gate. Check it before pushing with the same script CI runs:

```bash
.github/scripts/check_dco.sh origin/main HEAD
```

## Where to start reading

- [Architecture](../design/architecture.md) — how the subsystems fit together
- [Sequence-first](../design/sequence-first.md) — the one design decision that
  explains most of the interior
- [Contracts](../design/contracts.md) — what is frozen, and why
- [Adding a model](../design/adding-a-model.md) and
  [Adding a parallelism plan](../design/adding-a-parallelism-plan.md) — the two
  extensions that should never require touching core
