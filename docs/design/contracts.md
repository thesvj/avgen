# Contracts

avgen is built by several people working on several subsystems at once. That
only works if the interfaces between them are written down, frozen, and changed
deliberately. The normative document is
[`CONTRACTS.md`](https://github.com/thesvj/avgen/blob/main/CONTRACTS.md)
in the repository root; this page explains the reasoning.

## What is frozen

| Contract | Why it is frozen |
|---|---|
| `TokenStream`, `PatchLayout`, `TextContext` | Every model, every parallel plan, and the checkpoint layer index into these fields |
| `ModelInput` / `ModelOutput` | The model ABI. Two ABIs means two of everything downstream |
| `MediaBatch`, `MediaBatchSpec`, `ConditionMode` | The data-edge format, and it is on disk |
| `Patchifier` protocol | The grid ↔ sequence boundary |
| `RNGStreams` | Determinism depends on the stream layout being stable |
| `Stateful`, `TrainState` | DCP saves these without knowing what they are |
| `StepMetrics` | The telemetry contract |
| `ParallelDims` and the mesh dimension names | Plans, checkpoints, and the simulator all index the mesh by name |

Changing any of these requires an
[issue](https://github.com/thesvj/avgen/issues/new).

## The model ABI

```python
def forward(self, inputs: ModelInput) -> ModelOutput: ...
```

One argument, one return, both frozen dataclasses. Not `forward(video, audio,
text, timesteps, mask, positions, condition_mode, ...)`.

The reason is that a positional signature is a contract nobody can extend. Add
an argument and every caller breaks; add it with a default and the ordering
becomes a trap. `ModelInput` can gain a field without breaking any model that
does not read it, and `validate()` can check cross-field invariants — that the
audio stream's batch size matches the video's, that noise levels are in range,
that a conditioned token is not also masked out — in one place instead of at
every call site.

`ModelOutput` is in **token space**, not grid space: `(B, L, patch_dim)`. The
model never unpatchifies. That is the objective's job or the pipeline's, and
keeping it out of the model means the model does not need to know whether its
sequence is CP-sharded.

## Naming is part of the contract

A block's submodules must be named exactly:

```text
attention_norm, attention{.q_proj, .k_proj, .v_proj, .out_proj}
cross_norm,     cross_attention{...}          (optional)
ffn_norm,       feed_forward{.gate_proj, .up_proj, .down_proj}
```

and the root: `patch_embed`, `time_embed`, `text_proj`, `blocks`, `final_norm`,
`final_proj`.

This looks like unnecessary rigidity until you look at what addresses those
names: `standard_block_plan()` and `standard_root_plan()` build the
tensor-parallel plan by module path. A model that names its projection
`to_q` instead of `q_proj` does not get a clear error — it gets a plan that
silently does not apply to that module, and a job that runs correctly and
slowly.

Similarly, **a model must expose `.blocks` as an `nn.ModuleList`**. FSDP2
wrapping, activation checkpointing, and `torch.compile` all address the model
per block. Without that attribute, none of them work, and again the failure is
silent.

## The scale rules

`CONTRACTS.md` §6 lists twelve rules that exist because of scale. They share a
property: **each one passes every test you can run on a workstation and fails on
a cluster.**

1. **RNG varies on `data_rank` only.** CP/TP ranks holding shards of one sample
   must draw identical noise.
2. **Data shards on `data_rank` only.** Same reason.
3. **Losses and metrics reduce over `dp_cp`**, never the whole world. Reducing
   over the world divides by ranks holding the same loss.
4. **Never call `.item()` inside a training step.** A per-step host sync
   serialises every rank against the slowest.
5. **`sample_ids` stay on CPU.** They are never used in computation.
6. **Gradient clipping goes through `avgen.parallel.clip_grad_norm`**, which
   handles DTensor and pipeline stages.
7. **Optional deps import lazily**, with a `RuntimeError` naming the extra.
8. **Validation at boundaries, never in the hot loop.**
9. **A model exposes `.blocks: nn.ModuleList`.**
10. **Anything checkpointable implements `Stateful`.**
11. **No `torch.cuda` call at import time.** The package must import on a
    CPU-only machine — which is also what makes the simulator possible.
12. **Determinism:** the same seed, config, and rank count reproduce the same
    loss curve. No `set` iteration order in anything that affects computation.

Rules 1, 2, and 3 are the ones that produce wrong results rather than crashes,
which makes them the most expensive to find. They are on the pull-request
checklist for that reason.

## House style as contract

Some of the style rules are load-bearing:

- **`@dataclass(frozen=True, slots=True)` for config**, with validation in
  `__post_init__` raising `ValueError`/`TypeError` naming the field and the
  rejected value. Frozen because a step that mutates its config reproduces only
  under retry; the error message matters because it is read from a log with 1024
  ranks' worth of noise around it.
- **`__all__` in every module, sorted.** The public surface is explicit, and the
  API reference is generated from it.
- **Absolute imports only.** A relative import inside a subsystem makes moving a
  module a cross-subsystem change.
- **Comments explain why, never what.** This is a teaching codebase; the reader
  should finish a file understanding the domain, not just the code. A comment
  that restates the line below it is deleted in review.

## Changing a contract

1. Open an [issue](https://github.com/thesvj/avgen/issues/new)
   stating the problem first, the proposal written as it would appear in
   `CONTRACTS.md`, the alternatives and why each lost, the compatibility impact,
   and the migration.
2. Ten working days of comment.
3. Decision by maintainer consensus, or steering-group majority.
4. Implementation updates `CONTRACTS.md`, the docs, and `CHANGELOG.md` in one
   pull request.

If a contract genuinely blocks you, open the issue rather than editing it
locally. A locally-edited contract is a contract that two subsystems now
disagree about, and the disagreement will surface as a shape error at 512 ranks.
