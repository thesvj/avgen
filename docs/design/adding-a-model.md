# Adding a model

A new model is a pure addition: a file in `src/avgen/models/`, a registry
decoration, and a test. Nothing in `avgen.core` or `avgen.parallel` changes. If
you find yourself editing either, stop — either the registry is missing a hook,
or the change belongs elsewhere.

## The six requirements

### 1. Implement the ABI

```python
from avgen.core.model_input import ModelInput, ModelOutput


class MyDiT(nn.Module):
    def forward(self, inputs: ModelInput) -> ModelOutput: ...
```

One frozen dataclass in, one out. `ModelOutput` is in **token space** —
`(B, L, patch_dim)` — not grid space. The model never unpatchifies, which is
what lets it work unchanged whether or not the sequence is context-parallel
sharded.

### 2. Expose `.blocks` as an `nn.ModuleList`

```python
self.blocks = nn.ModuleList([MyBlock(config) for _ in range(config.depth)])
```

FSDP2 wrapping, activation checkpointing, and `torch.compile` all address the
model per block. Without this attribute none of them apply, and the failure is
silent — the job runs, unsharded and uncheckpointed, until it OOMs at a scale
where you assumed it would not.

### 3. Name submodules exactly

```text
attention_norm
attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj
cross_norm, cross_attention.{q_proj, k_proj, v_proj, out_proj}   (optional)
ffn_norm
feed_forward.gate_proj, feed_forward.up_proj, feed_forward.down_proj
```

Root: `patch_embed`, `time_embed`, `text_proj`, `blocks`, `final_norm`,
`final_proj`.

These are the paths `standard_block_plan()` and `standard_root_plan()` address.
A projection named `to_q` gets no error — it gets a tensor-parallel plan that
does not apply to it, and a job that is correct and slow.

### 4. Implement `TensorParallelizable`

```python
from avgen.parallel.tensor import standard_block_plan, standard_root_plan


def tensor_parallel_plan(self, *, sequence_parallel: bool) -> tuple[dict, dict]:
    return (
        standard_root_plan(sequence_parallel=sequence_parallel),
        standard_block_plan(sequence_parallel=sequence_parallel),
    )
```

If your block is standard, that is the whole implementation. If it is not, write
the plan explicitly — and read
[Adding a parallelism plan](adding-a-parallelism-plan.md) first.

### 5. Implement `model_shape`

```python
from avgen.simulate.memory import ModelShape


def model_shape(self, *, sequence_length: int, micro_batch_size: int = 1) -> ModelShape:
    return ModelShape(
        parameters=self.parameter_count(),
        depth=self.config.depth,
        width=self.config.width,
        num_heads=self.config.num_heads,
        mlp_ratio=self.config.mlp_ratio,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        text_tokens=self.config.text_tokens,
    )
```

This is what lets the simulator price your model. Without it, nobody can answer
"does this fit at 512 ranks" without launching, and the CI plan check cannot
cover your model.

Be honest about the numbers. A `parameters` count that excludes the embeddings
produces a memory estimate that is wrong in the direction that gets you OOMed.

### 6. Register it

```python
from avgen.models.registry import register_model


@register_model("my_dit")
class MyDiT(nn.Module): ...
```

`build_model("my_dit", config_mapping)` now works, and the name is usable from a
YAML config and from the CLI.

## Config

```python
@dataclass(frozen=True, slots=True)
class MyDiTConfig:
    depth: int = 32
    width: int = 2048
    num_heads: int = 16
    mlp_ratio: int = 4
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_tokens: int = 256

    def __post_init__(self) -> None:
        if self.width % self.num_heads != 0:
            raise ValueError(
                f"width={self.width} must be divisible by num_heads="
                f"{self.num_heads}; head_dim would be fractional"
            )
```

Frozen, slotted, validated in `__post_init__`, with the field name and the
rejected value in every error message. That message will be read from a log with
1023 other ranks' output around it.

## Working with `TokenStream` inside the model

```python
def forward(self, inputs: ModelInput) -> ModelOutput:
    stream = inputs.video
    hidden = self.patch_embed(stream.tokens)  # (B, L, W)
    time = self.time_embed(stream.expanded_noise())  # per-sample or per-token
    positions = stream.coords  # (B, L, 3) seconds, row, col
```

Three things to get right:

- **Use `stream.coords` for positional encoding**, not the token index. That is
  what makes the model correct under CP sharding, packing, and variable
  resolution. If you index by position in the tensor, your model breaks the first
  time someone shards it — silently.
- **Respect `stream.mask`.** Padding tokens must not contribute to attention or
  to the loss. `loss_mask()` combines the validity mask with `conditioned` so
  clean anchors are excluded too.
- **Use `expanded_noise()`** rather than branching on whether `noise_level` is
  per-sample or per-token. That branch is how inpainting and continuation break.

## Audio-video models

Subclass the video model and add the audio stream and the fusion:

```python
@register_model("my_av_dit")
class MyAVDiT(MyDiT):
    def forward(self, inputs: ModelInput) -> ModelOutput: ...
```

`inputs.audio` is always present and may be zero-length; it is never `None`.
That is deliberate — an optional field means every call site branches, and one
of them will forget.

Audio is a temporal-only layout (`PatchLayout.temporal(...)`,
`is_temporal_only()` true), so the same patchifier machinery applies with
`height = width = 1`.

## Tests

Three, at minimum:

**A CPU smoke test with tiny shapes.**

```python
def test_forward_shapes():
    model = MyDiT(MyDiTConfig(depth=2, width=64, num_heads=4))
    output = model(tiny_model_input())
    output.validate(inputs)
```

**A simulator test at a large fake world size**, which is where plan bugs
actually live:

```python
def test_plan_applies_at_1024():
    dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
    report = simulate_rank(
        dims,
        build=lambda: MyDiT(config),
        apply_plan=lambda m, d, mesh: parallelize(m, d, mesh=mesh).model,
    )
    assert report["shard_efficiency"] > 0.9
```

**A shape test**, checking that `model_shape` agrees with the real parameter
count.

## Checklist

- [ ] `forward(ModelInput) -> ModelOutput`, output in token space
- [ ] `.blocks` is an `nn.ModuleList`
- [ ] Submodule names match the contract exactly
- [ ] `tensor_parallel_plan` implemented
- [ ] `model_shape` implemented and honest
- [ ] `@register_model("...")`
- [ ] Config is a frozen, slotted, validated dataclass
- [ ] Positional encoding reads `coords`, not the token index
- [ ] `mask` respected in attention and loss
- [ ] `__all__` sorted; Google docstrings; comments explain *why*
- [ ] CPU smoke test, simulator test, shape test
- [ ] Documented, and added to `CHANGELOG.md`

## Further reading

- [Sequence-first](sequence-first.md)
- [Contracts](contracts.md)
- [Adding a parallelism plan](adding-a-parallelism-plan.md)
