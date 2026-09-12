# Contributing

Patches are welcome. This is a young project, so the process is short.

## Setting up

avgen uses [uv](https://docs.astral.sh/uv/). A reproducible lockfile is the only
way a distributed bug report is actionable, so that part is not optional.

```bash
git clone https://github.com/thesvj/avgen
cd avgen
make dev          # every extra, dev + docs groups, pre-commit hooks
```

If you are only touching the core, a lighter install keeps the import surface
honest:

```bash
uv sync --group dev
```

You do not need a GPU for any of this. Everything except the `gpu` and
`multigpu` markers runs on CPU, and that includes the whole parallelism
simulator, so you can review a 1024-GPU plan from a laptop.

## The loop

```bash
make test-fast    # run this before you push
make lint         # ruff check
make format       # ruff format
make type         # mypy --strict
make test         # everything CI runs
make docs         # mkdocs build --strict
make simulate     # reference parallelism plans at 8, 64, 512, 1024 ranks
```

`make test-fast` skips the `slow`, `gpu`, `multigpu` and `optional_deps`
markers. On sixteen cores it takes about a minute and a half for roughly 1,520
tests, most of which is real forward and backward passes rather than overhead.

GPU tests run nightly on a self-hosted runner and are not a pull-request gate,
so a change that only works on your workstation can still land. Please do not
let that happen.

## House style

`ruff` and `mypy --strict` enforce most of it; `pyproject.toml` has the exact
configuration. The parts a linter cannot check:

- **Comments explain why, not what.** Every magic constant, ordering
  requirement and numerical choice should carry a line or two saying what
  breaks without it. A comment that just restates the line below it will be
  asked for in review, to be deleted.
- **Docstrings on classes that encode a design decision should say what the
  alternative was and why it lost.**
- Absolute imports only, and `from __future__ import annotations` at the top of
  every module.
- Config is a frozen dataclass, validated in `__post_init__`, raising
  `ValueError` or `TypeError` that names the field and the rejected value.
- No `print`. Use the telemetry logger.
- Line length 88, Google-style docstrings.

### Optional dependencies

Never import an optional dependency at module scope. Import it inside the
function that needs it, and raise a `RuntimeError` naming the extra:

```python
def load_t5(name: str) -> nn.Module:
    try:
        from transformers import T5EncoderModel
    except ImportError as exc:
        raise RuntimeError(
            "the text tower needs the 'text' extra: pip install 'avgen[text]'"
        ) from exc
    return T5EncoderModel.from_pretrained(name)
```

This is not about tidiness. A cluster image that pulls `diffusers` into every
rank's import path costs startup time on 1024 ranks, and drags a whole
transitive dependency graph into a job that never uses it.

## Extending avgen without touching core

The registries exist so that the usual extensions are pure additions. If you
find yourself editing `src/avgen/core/` or `src/avgen/parallel/` to add a
feature, please stop and open an issue instead. Either the registry is missing a
hook, or the change belongs somewhere else.

You need not contribute the extension at all. Each registry also reads an entry
point group, so it can live in your own package:

| Group | Resolves to |
|---|---|
| `avgen.models` | an `nn.Module` subclass |
| `avgen.metrics` | a `RunningMetric` class |
| `avgen.samplers` | a callable taking a `SamplerConfig` |
| `avgen.rewards` | a callable returning a `RewardModel` |

### A new model

1. Subclass `nn.Module`, take a frozen dataclass config as the first
   constructor argument, and implement
   `forward(inputs: ModelInput) -> ModelOutput`.
2. Expose `.blocks` as an `nn.ModuleList`. FSDP2 wrapping, activation
   checkpointing and `torch.compile` all address the model per block, and none
   of them work without this attribute.
3. Implement
   `tensor_parallel_plan(*, sequence_parallel) -> (root_plan, block_plan)`, or
   use `standard_block_plan()` and `standard_root_plan()`. Those need the
   submodule names in `CONTRACTS.md` §4 exactly.
4. Implement `model_shape(*, sequence_length, micro_batch_size)` so the
   simulator can price your model.
5. Decorate with `@register_model("your_name")`.
6. Add a CPU smoke test at tiny shapes, and a simulator test that applies the
   plan at a large fake world size.

### A new data source

Implement `avgen.data.protocols.DataSource`: `__iter__` yielding `MediaBatch`,
plus `state_dict()` and `load_state_dict()`. Resumability is not optional. A
source that cannot restore its cursor turns every preemption into lost samples
and a silently different data order.

Shard on `data_rank` only. Context-parallel and tensor-parallel ranks hold
shards of the *same* sample and must receive identical data.

### A new example or benchmark

`examples/` holds complete, runnable programs, not fragments, and every one is
executed by `tests/test_examples.py`. An example that no longer runs is worse
than no example at all: it is the first code a new user copies, and it fails on
their machine instead of in CI. An example must run on CPU in seconds with no
dataset and no network, and must be listed in `examples/README.md`.

`benchmarks/` keeps the simulator honest by measuring the real thing. A
benchmark reports a median and a spread rather than one number, discards warmup
explicitly, and names the hardware and shapes in its own output.

## Before you open a pull request

**If you touched anything parallelism-related**, run the simulator and paste the
before/after table into the pull request:

```bash
make simulate
```

This matters more than it sounds. Any change to `src/avgen/parallel/**`, to a
model's `tensor_parallel_plan`, or to its `model_shape` changes behaviour at
scales none of us can test interactively. Without the table, a reviewer cannot
tell whether you moved predicted MFU at 1024 ranks by 1% or by 30%.

Otherwise, the usual things: one concern per pull request, a title in the
imperative prefixed by subsystem (`parallel: shard rotary tables under context
parallelism`), a body explaining why, a test that fails without your fix, and a
`CHANGELOG.md` entry under `## [Unreleased]` if a user would notice.

Draft pull requests are welcome for anything large. Please open one early with a
sketch, rather than landing three weeks of work nobody has agreed with yet.

## The rules that only bite at scale

From `CONTRACTS.md` §6. Break one of these and you get a bug that passes every
test you can run at your desk, and then fails on a cluster:

1. RNG varies on `data_rank` only.
2. Data shards on `data_rank` only.
3. Losses and metrics reduce over the `dp_cp` mesh, never the whole world.
4. Never call `.item()` inside a training step.
5. `sample_ids` stay on CPU.
6. Gradient clipping goes through `avgen.parallel.clip_grad_norm`.
7. Optional dependencies import lazily.
8. Validation happens at boundaries, never in the hot loop.
9. A model exposes `.blocks: nn.ModuleList`.
10. Anything checkpointable implements `Stateful`.
11. No `torch.cuda` call at import time.
12. The same seed, config and rank count reproduce the same loss curve.

Please check these yourself before asking for a review.

## Reporting bugs

Use the [bug report form](https://github.com/thesvj/avgen/issues/new?template=bug_report.yml).
It asks for the avgen version, torch version, GPU, world size, parallelism
degrees and the exact config, because a distributed training bug is not
reproducible without all six.

Two commands resolve a good share of reports before they are filed:

```bash
avgen info                                        # install, imports, hardware
avgen simulate --config your_config.yaml --world-size N
```

The first catches a broken install or an invisible GPU. The second catches a
configuration that was never going to fit, which at runtime just looks like an
OOM of mysterious origin.

For anything security-related, see [`SECURITY.md`](SECURITY.md) instead of the
issue tracker.
