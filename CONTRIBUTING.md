# Contributing to avgen

avgen is a teaching codebase as much as a training framework. A patch that works
but leaves the next reader guessing *why* is not finished. Read
[`CONTRACTS.md`](CONTRACTS.md) before your first change — it is the interface
agreement between subsystems, and it is normative.

---

## Development setup

avgen uses [uv](https://docs.astral.sh/uv/) for environment management. Nothing
else is supported, because a reproducible lockfile is the only way a distributed
bug report is actionable.

```bash
git clone https://github.com/avgen-project/avgen
cd avgen
make dev          # uv sync --all-extras --group dev --group docs + pre-commit install
```

`make dev` installs every optional extra. If you are only touching core, the
lighter install is enough and keeps the import surface honest:

```bash
uv sync --group dev
```

The core package depends on **torch, numpy, pyyaml, safetensors and nothing
else**. If your change makes `import avgen` require anything more, it is wrong.
See [Optional dependencies](#optional-dependencies).

### GPU is not required

Everything except the `gpu`/`multigpu` test markers runs on CPU, including the
whole parallelism simulator. You can develop, review a 1024-GPU parallelism
plan, and land a change without ever touching a cluster.

---

## The loop

```bash
make test-fast    # the subset that must pass before you push
make lint         # ruff check
make format       # ruff format
make type         # mypy --strict on src/avgen
make test         # everything not marked gpu/multigpu
```

`make test-fast` deselects `slow`, `gpu`, `multigpu` and `optional_deps`, runs
under `pytest -x -q -n auto`, and should finish in well under a minute. It is
the gate for "am I about to waste a CI run". `make test` is what CI runs.

Markers, defined in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `gpu` | Needs at least one CUDA device |
| `multigpu` | Needs at least two CUDA devices |
| `slow` | Excluded from the fast subset |
| `optional_deps` | Needs an optional extra installed |

GPU-marked tests run nightly on a self-hosted runner
(`.github/workflows/gpu-nightly.yml`). They are not a pull-request gate, so a
change that only works on your workstation can still land — do not let it.

---

## House style

Enforced mechanically by `ruff` and `mypy --strict`; the exact configuration
lives in `pyproject.toml`. The parts that a linter cannot check:

- **Comments explain why, never what.** Every magic constant, ordering
  requirement, and numerical choice carries one to three lines saying what
  breaks without it. A comment that restates the line below it will be asked
  for in review — to be deleted.
- **Docstrings on classes that encode a design decision say what the
  alternative was and why it lost.** `TokenStream` does not just say "holds
  tokens"; it says why the model interior is a sequence and not a grid.
- **Absolute imports only.** Relative imports are a lint error.
- **`from __future__ import annotations` at the top of every module.**
- **Config is a frozen dataclass** with validation in `__post_init__` that
  raises `ValueError`/`TypeError` naming the field and the rejected value.
- **No `print`.** Use the telemetry logger.

Line length is 88. Docstrings are Google style with `Args:`, `Returns:`,
`Raises:`.

### Optional dependencies

Never import an optional dependency at module scope. Import it inside the
function that needs it and raise a `RuntimeError` naming the extra:

```python
def load_t5(name: str) -> nn.Module:
    try:
        from transformers import T5EncoderModel
    except ImportError as exc:  # pragma: no cover - exercised by a marked test
        raise RuntimeError(
            "the text tower needs the 'text' extra: pip install 'avgen[text]'"
        ) from exc
    return T5EncoderModel.from_pretrained(name)
```

This is not politeness. A cluster image that pulls `diffusers` into every rank's
import path costs startup time on 1024 ranks and drags a transitive dependency
graph into a job that never used it.

---

## Extending avgen without touching core

The registries exist so that the common extensions are pure additions. If you
find yourself editing `src/avgen/core/` or `src/avgen/parallel/` to add a
feature, stop and open an issue — either the registry is missing a hook, or the
change belongs in a different place.

### A new model

1. Subclass `nn.Module`, take a frozen dataclass config, and implement
   `forward(inputs: ModelInput) -> ModelOutput`. `ModelInput`/`ModelOutput` are
   the model ABI; do not invent a second one.
2. Expose `.blocks` as an `nn.ModuleList`. FSDP2 wrapping, activation
   checkpointing, and `torch.compile` all address the model per block, and none
   of them work without this attribute.
3. Implement `tensor_parallel_plan(*, sequence_parallel) -> (root_plan, block_plan)`,
   or use `standard_block_plan()` / `standard_root_plan()` — which requires the
   submodule names in `CONTRACTS.md` §4 exactly (`attention.q_proj`,
   `feed_forward.gate_proj`, and so on).
4. Implement `model_shape(*, sequence_length, micro_batch_size)` returning an
   `avgen.simulate.memory.ModelShape`, so the simulator can price your model.
5. Decorate with `@register_model("your_name")`.
6. Add a CPU smoke test with tiny shapes, and a simulator test that applies the
   parallel plan at a large fake world size.

### A new dataset or data source

Implement `avgen.data.protocols.DataSource`: `__iter__` yielding `MediaBatch`,
plus `state_dict()`/`load_state_dict()`. **Resumability is not optional** — a
source that cannot restore its cursor turns every preemption into lost samples
and a silently different data order.

Shard on `data_rank` only. Context-parallel and tensor-parallel ranks hold
shards of the *same* sample and must receive identical data; see
`ParallelDims.data_coordinates`.

### A new parallelism plan

Read [`docs/design/adding-a-parallelism-plan.md`](docs/design/adding-a-parallelism-plan.md).
The short version: a plan is a function from `(model, ParallelDims, DeviceMesh)`
to a transformed model, applied by `avgen.parallel.apply.parallelize`. It must
be idempotent under a re-apply, must not assume a GPU is present, and must be
exercised by a simulator test — see the next section.

### A new metric

Implement `avgen.eval.Metric` (`update`/`compute`/`reset`), register it with
`register_metric(name)`, and make the built-in path work with **no optional
dependencies**. A metric that needs a gated backend declares it and raises a
clear error naming the extra when it is missing.

---

## Run the simulator before proposing a parallelism change

This is the single most important local step in this project, and it is the
reason a parallelism change can be reviewed at all.

Any change that touches `src/avgen/parallel/**`, a model's
`tensor_parallel_plan`, or a model's `model_shape` changes the behaviour of
plans at scales none of us can test interactively. Before you open the pull
request:

```bash
make simulate     # reference plans at world sizes 8, 64, 512, 1024
```

This runs `.github/scripts/check_reference_plans.py`, which builds a real
`DeviceMesh` for each world size under `FakeProcessGroup`, applies the plan,
and prices the result with the memory, communication, and compute models. It
takes a few seconds and needs no GPU. CI runs exactly the same script
(`.github/workflows/simulate.yml`), so a green local run means a green CI run.

**Paste the before/after table into your pull request.** A reviewer cannot
otherwise tell whether your change moved predicted MFU at 1024 ranks by 1% or
by 30%.

To explore rather than check, use the CLI:

```bash
uv run avgen plan --model 2b --world-size 512 --seq-len 65536
uv run avgen simulate --config configs/your_run.yaml --world-size 1024
```

What the simulator is exact about: shapes, sharding, memory accounting,
collective counts and sizes. What it estimates: time. What it says nothing
about: numerics — fake collectives return uninitialised data, so a simulated
loss is meaningless.

---

## Pull requests

### Sign your commits (DCO)

avgen uses the [Developer Certificate of Origin](https://developercertificate.org/).
Every commit must carry a `Signed-off-by` line matching the author:

```bash
git commit -s -m "parallel: keep tp innermost when pp is enabled"
```

`git commit -s` adds it for you. To fix an unsigned branch:

```bash
git rebase --signoff main
```

There is no CLA. Signing off means you certify you wrote the change, or have
the right to submit it under Apache-2.0.

### What a reviewable pull request looks like

- **One concern per PR.** A refactor plus a behaviour change is two PRs.
- **The title says what changed, in the imperative**, prefixed by subsystem:
  `parallel: shard rotary tables under context parallelism`.
- **The body says why.** What was broken or missing, what you chose, what you
  rejected and why. If a number changed, show the number.
- **Tests.** A bug fix comes with the test that fails without it. A feature
  comes with a CPU smoke path.
- **Docs.** A behaviour change updates the relevant page under `docs/` in the
  same PR. A new public symbol needs a docstring; the API reference is
  generated, so that is all it needs.
- **`CHANGELOG.md`** gets an entry under `## [Unreleased]` for anything a user
  would notice.
- **Contract changes** (`CONTRACTS.md`) are a separate, explicit discussion —
  see the RFC process in [`GOVERNANCE.md`](GOVERNANCE.md).

Draft PRs are welcome and encouraged for anything large. Open one early with a
sketch rather than landing three weeks of work nobody agreed with.

### The rules that only bite at scale

These come from `CONTRACTS.md` §6. Violating one produces a bug that passes
every test you can run on your desk and fails on a cluster:

1. RNG varies on `data_rank` only.
2. Data shards on `data_rank` only.
3. Losses and metrics reduce over the `dp_cp` mesh, never the whole world.
4. Never call `.item()` inside a training step.
5. `sample_ids` stay on CPU.
6. Gradient clipping goes through `avgen.parallel.clip_grad_norm`.
7. Optional deps import lazily.
8. Validation at boundaries, never in the hot loop.
9. A model exposes `.blocks: nn.ModuleList`.
10. Anything checkpointable implements `Stateful`.
11. No `torch.cuda` call at import time.
12. The same seed, config, and rank count reproduce the same loss curve.

Reviewers check these explicitly. Please check them yourself first.

---

## Reporting bugs

Use the [bug report form](https://github.com/avgen-project/avgen/issues/new?template=bug_report.yml).
It asks for the avgen version, torch version, GPU, world size, parallelism
degrees, and the exact config, because a distributed training bug is not
reproducible without all six.

Security issues do **not** go in the issue tracker — see
[`SECURITY.md`](SECURITY.md).

---

## Code of conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
