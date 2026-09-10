# avgen — internal build contract

**Everyone working on this repo reads this file first.** It is the interface
agreement between subsystems. If you need something that is not here, add it
here in the same change, do not invent it locally.

---

## 1. Ground truth

The authoritative contracts are the source files, already written and frozen:

- `src/avgen/core/` — tensors, batch, tokens, patchify, model_input, rng, state, metrics
- `src/avgen/parallel/` — dims, env, comm, fsdp, tensor, context, pipeline, activation, precision, apply
- `src/avgen/simulate/` — world, memory, comms, compute, plan, chakra

**Read the ones you depend on before writing anything.** Do not modify a file
you do not own (see §5). If a core contract genuinely blocks you, say so in your
report rather than editing it.

---

## 2. House style — non-negotiable

Enforced by `ruff` and `mypy --strict` in CI (`pyproject.toml` has the exact config).

- Python ≥ 3.11. `from __future__ import annotations` at the top of every module.
- **Absolute imports only.** `from avgen.core.tokens import TokenStream`. Relative imports are banned by lint.
- **Full type annotations** on every public function, method, and attribute. `mypy --strict` must pass.
- **Google-style docstrings** on every public module, class, function, and method, with `Args:`, `Returns:`, `Raises:`.
- `__all__` in every module, sorted.
- Line length 88.
- Public API has no leading underscore; internals do. A module that is entirely internal is named `_something.py`.
- **Dataclasses** for config: `@dataclass(frozen=True, slots=True)`, validation in `__post_init__`, raise `ValueError`/`TypeError` with the field name and the rejected value in the message.
- No `print`. Use the telemetry logger.
- No optional dependency imported at module scope. Import it inside the function that needs it and raise a `RuntimeError` naming the extra (`pip install 'avgen[codecs]'`) if it is missing.

### Comment discipline

Comments explain **why**, never what. Every non-obvious decision — a magic
constant, an ordering requirement, a numerical choice — carries a one-to-three
line comment saying what breaks without it. Docstrings on classes that encode a
design decision should say what the alternative was and why it lost. This is a
teaching codebase; the reader should finish a file understanding the domain, not
just the code.

Do not write a comment that restates the line below it.

---

## 3. Frozen core contracts (do not redefine these)

```python
# avgen.core.tokens
@dataclass(frozen=True, slots=True)
class PatchLayout:
    frames: int; height: int; width: int
    patch_frames: int = 1; patch_height: int = 2; patch_width: int = 2
    channels: int = 1
    # properties: grid -> (gf, gh, gw), num_tokens, patch_dim, is_temporal_only
    # classmethods: temporal(frames, channels, patch_frames), empty()

@dataclass(frozen=True, slots=True)
class TokenStream:
    tokens: Tensor        # (B, L, W)   sequence-major; L is the CP-shard axis
    coords: Tensor        # (B, L, 3)   float32 (seconds, row, col) PHYSICAL units
    mask: Tensor          # (B, L)      bool validity
    noise_level: Tensor   # (B,) or (B, L)  float32 in [0,1]; 0 = clean
    layout: PatchLayout   # static geometry, survives CP sharding
    conditioned: Tensor   # (B, L)      bool clean-anchor mask
    # properties: batch_size, length, width, device, is_empty, per_token_noise
    # methods: expanded_noise(), with_tokens(t), masked(), loss_mask(), validate()
    # classmethod: empty_like(batch, width, device=, dtype=)

@dataclass(frozen=True, slots=True)
class TextContext:
    features: Tensor      # (B, T, W)
    mask: Tensor          # (B, T) bool
    # properties: is_empty ; methods: validate(), nullified()
    # classmethod: empty(batch, width, device=, dtype=)
```

```python
# avgen.core.model_input  — THE model ABI. Every model: forward(ModelInput) -> ModelOutput
@dataclass(frozen=True, slots=True)
class ModelInput:
    video: TokenStream
    audio: TokenStream          # may be zero-length (T2V); never None
    text: TextContext           # may be empty
    condition_mode: Tensor      # (B,) int64, ConditionMode
    # properties: batch_size, device, has_audio, has_text, total_tokens
    # methods: replace_streams(video=, audio=), unconditional(), validate()

@dataclass(frozen=True, slots=True)
class ModelOutput:
    video: Tensor               # (B, L_v, patch_dim)  TOKEN SPACE, not grid
    audio: Tensor               # (B, L_a, patch_dim)
    auxiliary: tuple[Tensor, ...] = ()
    # method: validate(inputs)
```

```python
# avgen.core.batch
class ConditionMode(IntEnum):
    JOINT=0; VIDEO_TO_AUDIO=1; AUDIO_TO_VIDEO=2; VIDEO_ONLY=3
    CONTINUATION=4; INPAINT=5; IMAGE_TO_VIDEO=6; VIDEO_TO_VIDEO=7

@dataclass(frozen=True, slots=True)
class MediaBatchSpec:   # dense, data-edge only
    schema_version, bucket_id: int
    video_shape: tuple  # (B, C, T, H, W)
    audio_shape: tuple  # (B, C, T)  T may be 0
    text_shape: tuple   # (B, T, W)  T may be 0
    video_timebase_num/den, audio_timebase_num/den: int   # exact rationals
    video_codec_id, audio_codec_id: str
    target_spec: TensorBundleSpec
    # properties: batch_size, video_fps, audio_fps, has_audio, has_text,
    #             video_tokens, audio_tokens, sequence_length
    # to_dict()/from_dict()

@dataclass(frozen=True, slots=True)
class MediaBatch:
    video, audio, text, video_mask, audio_mask,
    video_positions, audio_positions, sample_ids, spec, text_mask, targets
    # properties: device ; methods: to(device), validate()
    # sample_ids is int64 and STAYS ON CPU

def stack_batches(batches) -> MediaBatch
def null_text_conditioning(text, text_mask) -> (Tensor, Tensor)
```

```python
# avgen.core.patchify
class Patchifier(Protocol):
    def to_tokens(self, grid, *, positions, mask, noise_level, conditioned=None) -> TokenStream
    def to_grid(self, stream: TokenStream) -> Tensor
    def layout_for(self, grid_shape: tuple[int, ...]) -> PatchLayout

@dataclass(frozen=True, slots=True)
class GridPatchifier:      # the default
    patch_frames: int = 1; patch_height: int = 2; patch_width: int = 2
    normalize_space: bool = False

def patchify_grid(grid, layout) -> Tensor      # time-major, then row, then col
def unpatchify_grid(tokens, layout) -> Tensor  # exact inverse
def build_temporal_coords(positions, layout, *, normalize_space=False) -> Tensor
def build_spatial_coords(layout, *, batch, fps, device=, start_seconds=0.0) -> Tensor
```

```python
# avgen.core.rng
class RNGStreams:
    noise, timestep, conditioning, sampler: torch.Generator
    @classmethod from_seed(seed, *, device="cpu")
    @classmethod for_rank(seed, *, data_rank, device="cpu")   # varies on DATA rank ONLY
    fork(purpose) -> Generator ; state_dict() ; load_state_dict() ; validate()
```

```python
# avgen.core.state
class Stateful(Protocol):   state_dict() ; load_state_dict(state)
class LRSchedule(Stateful): step() ; get_last_lr() -> list[float]
class EMA(Stateful):        update(model)
class DataCursor(Stateful): advance(samples)

@dataclass(slots=True)
class TrainState:
    model: nn.Module; optimizer: Optimizer; rng: RNGStreams
    schedule: LRSchedule | None = None; ema: EMA | None = None
    step: int = 0; samples_seen: int = 0; tokens_seen: int = 0; epoch: int = 0
    extras: dict[str, Stateful]
    # validate(), progress() -> dict[str,int], load_progress(mapping)
```

```python
# avgen.core.metrics
@dataclass(frozen=True, slots=True)
class StepMetrics:
    loss, video_loss, audio_loss: Tensor        # fp32 scalar, detached
    valid_video_tokens, valid_audio_tokens: Tensor   # int64 scalar
    grad_norm: Tensor                            # fp32 scalar
    nonfinite, skipped: Tensor                   # bool scalar
    # validate(), to_mapping() -> dict[str,float]  (ONE sync; call at log cadence)
    # classmethod zeros(device)
```

```python
# avgen.parallel  (all already written)
ParallelDims(world_size, dp_replicate=1, dp_shard=-1, tensor=1, context=1, pipeline=1)
  .build_mesh(device_type) -> DeviceMesh   # dims: pp, dp_replicate, dp_shard, cp, tp
                                           # + flattened: dp_shard_cp, dp_cp
  .data_coordinates(mesh) -> (data_rank, data_world)
  .sequence_coordinates(mesh) -> (shard_index, shard_count)
  .gradient_accumulation_for(global_batch_size=, local_batch_size=) -> int
  .dp_size, .sequence_shard_size, .model_shard_size, .describe()
  .{dp,tp,cp,pp}_enabled, .dp_shard_enabled, .dp_replicate_enabled

parallelize(model, dims, mesh=None, config=ParallelConfig()) -> ParallelModel
ParallelModel(.model, .dims, .mesh, .applied, .submesh(name), .cp_mesh, .pp_mesh)
ParallelConfig(precision, activation_checkpoint, fsdp, compile_blocks, compile_mode,
               block_attribute="blocks", sequence_parallel=True)

init_distributed() -> DistributedEnv(.rank,.local_rank,.world_size,.local_world_size,
                                     .backend,.device,.is_main,.is_local_main,
                                     .is_distributed,.num_nodes)
shutdown_distributed(); barrier(env); unwrap_model(m); local_rank_device()
data_mesh(mesh); all_reduce_{sum,mean,max}(t, mesh); clip_grad_norm(params, max_norm, pp_mesh=)
broadcast_object(obj, src=0); gather_object(obj, dst=0)
shard_stream(stream, cp_mesh); gather_stream(stream, cp_mesh); pad_to_multiple(stream, n)
context_parallel_region(cp_mesh, buffers=, buffer_seq_dims=)
TensorParallelizable protocol: tensor_parallel_plan(*, sequence_parallel) -> (root_plan, block_plan)
standard_block_plan(...) ; standard_root_plan(...)
PrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32", ...)
ActivationCheckpointConfig(mode="selective_op", layer_interval=2, save_op_frequency=1)
```

---

## 4. Contracts your subsystem must provide

These are the APIs other agents will import. Implement them exactly.

### `avgen.models`
```python
# registry.py
def register_model(name: str) -> Callable[[type], type]      # decorator
def build_model(name: str, config: Mapping[str, Any]) -> nn.Module
def list_models() -> tuple[str, ...]
def model_config_class(name: str) -> type

# dit.py  — must subclass nn.Module, implement TensorParallelizable,
#           expose `.blocks` as an nn.ModuleList (parallel layer requires this)
@register_model("video_dit")
class VideoDiT(nn.Module):
    def __init__(self, config: VideoDiTConfig) -> None
    def forward(self, inputs: ModelInput) -> ModelOutput
    def tensor_parallel_plan(self, *, sequence_parallel: bool) -> tuple[dict, dict]
    @property
    def patchifier(self) -> Patchifier
    def parameter_count(self) -> int
    def model_shape(self, *, sequence_length: int, micro_batch_size: int = 1) -> ModelShape
    # ^ returns avgen.simulate.memory.ModelShape so the simulator can price it

@register_model("av_dit")
class AVDiT(VideoDiT): ...   # adds the audio stream + AV fusion

# Submodule names inside a block MUST be:
#   attention_norm, attention{.q_proj,.k_proj,.v_proj,.out_proj}
#   cross_norm, cross_attention{.q_proj,...}      (optional)
#   ffn_norm, feed_forward{.gate_proj,.up_proj,.down_proj}
# Root: patch_embed, time_embed, text_proj, blocks, final_norm, final_proj
# These names are what standard_block_plan()/standard_root_plan() address.
```

### `avgen.train`
```python
# timestep.py
class TimestepSampler(Protocol):
    def sample(self, batch: int, *, device, generator, sequence_length: int | None = None) -> Tensor
@register_timestep_sampler("uniform"|"logit_normal"|"shifted_logit_normal"|"mode")
# shifted_logit_normal MUST implement resolution/sequence-length-dependent shift
#   shift(L) interpolated between (base_len, base_shift) and (max_len, max_shift)
#   t' = shift*t / (1 + (shift-1)*t)

# conditioning.py
class ConditioningSampler(Protocol):
    def sample(self, batch: MediaBatch, *, rng: RNGStreams) -> ConditioningPlan
@dataclass ConditioningPlan: condition_mode, video_conditioned, audio_conditioned, drop_text

# objective.py
class Objective(Protocol):
    def __call__(self, model, batch: MediaBatch, rng: RNGStreams, *,
                 patchifier: Patchifier, cp_mesh=None) -> ObjectiveOutput
@dataclass ObjectiveOutput: loss, video_loss, audio_loss (fp32 scalars)
class FlowMatchingObjective:   # rectified flow, velocity target v = noise - clean
    def __init__(self, config: FlowMatchingConfig, timestep_sampler, conditioning_sampler)

# ema.py — MUST be DTensor/FSDP-safe and support bf16 storage + sharded state
class ShardedEMA:  # implements avgen.core.EMA
    def __init__(self, model, *, decay=0.9999, storage_dtype=torch.bfloat16, warmup_steps=0)

# schedule.py
def build_schedule(name, optimizer, *, total_steps, warmup_steps, **kw) -> LRSchedule
# names: "constant","cosine","linear","wsd" (warmup-stable-decay)

# optimizer.py
def build_optimizer(model, *, name="adamw", lr, weight_decay, betas, eps, fused=None,
                    no_decay_patterns=("norm","bias","scale_shift","embedding")) -> Optimizer

# trainer.py
@dataclass TrainerConfig: ...
class Trainer:
    def __init__(self, state: TrainState, objective, parallel: ParallelModel, config, ...)
    def train_step(self, batch: MediaBatch, *, microbatch_index: int, accumulation: int) -> StepMetrics
    def fit(self, loader, *, total_steps: int) -> None
# functional core stays available:
def train_step(state, batch, objective, *, patchifier, gradient_accumulation_steps=1,
               microbatch_index=0, max_grad_norm=None, cp_mesh=None, pp_mesh=None)
    -> tuple[TrainState, StepMetrics]
```

### `avgen.data`
```python
# protocols.py
class DataSource(Protocol):
    def __iter__(self) -> Iterator[MediaBatch]
    def state_dict(self)/load_state_dict(state)      # resumable
class Bucket / BucketPlan                            # resolution x duration
# synthetic.py: SyntheticSource — deterministic, no deps, used by tests + smoke runs
# shard.py: write_shard/read_shard — mmap + safetensors, sha256 manifest, atomic COMMIT
# bucket.py: BucketSampler — resolution/duration buckets, aspect-ratio groups
# loader.py: build_loader(...) -> DataSource sharded by (data_rank, data_world)
#   MUST shard on data_rank ONLY (CP/TP ranks get identical data)
```

### `avgen.checkpoint`
```python
save(path, state: TrainState, *, parallel: ParallelModel, async_save=True, extras=None)
load(path, state: TrainState, *, parallel: ParallelModel) -> None
# DCP-based: reshards across a different rank count. async_save via dcp.async_save.
export_safetensors(path, model, *, dtype=None, metadata=None)   # release artifact
export_huggingface(path, model, ...)                            # dcp HuggingFaceStorageWriter
class CheckpointManager:  # retention, keep_last_n, keep_every, resume-latest
```

### `avgen.telemetry`
```python
class Logger(Protocol): log_metrics(dict, step) ; log_config(dict) ; close()
build_logger(names, ...) -> Logger      # "console","jsonl","tensorboard","wandb","noop"
class MetricAccumulator   # on-device accumulation, ONE sync at log cadence
class ThroughputMeter     # tokens/s, samples/s, MFU, step time p50/p99
class MemoryReporter      # peak, reserved, fragmentation, OOM-risk warning
profile_step(...)         # torch.profiler wrapper
```

### `avgen.infer`
```python
@dataclass SamplerConfig / class Sampler(Protocol): step(...)
"euler", "heun", "dpmpp_2m", "res_multistep"  registered samplers
class GuidanceConfig / apply_guidance(...)  # CFG, CFG-rescale, modality-CFG, APG
class GenerationPipeline:  # text -> latents -> media; shares ModelInput with training
    def __call__(self, prompts, *, steps, guidance, seed, ...) -> GeneratedMedia
```

### `avgen.codecs`
```python
class VideoCodec(Protocol): encode(pixels) -> latents ; decode(latents) -> pixels ; fingerprint
class AudioCodec(Protocol) / class TextEncoder(Protocol): encode(prompts) -> (features, mask)
# Identity/reference implementations with NO optional deps for tests.
# diffusers/transformers adapters imported lazily inside functions.
```

### `avgen.finetune`
```python
class LoRAConfig / apply_lora(model, config) -> nn.Module  # + DoRA flag
merge_lora(model) ; save_adapter(path, model) ; load_adapter(path, model)
freeze_except(model, patterns) ; class ControlAdapter  # ControlNet-style side tower
```

### `avgen.rl`
```python
# grpo.py: Flow-GRPO style. ODE->SDE conversion, group-relative advantage,
#          ratio clipping, KL-to-reference. MixGRPO window option.
class GRPOConfig / class GRPOTrainer
# dpo.py: Diffusion-DPO / flow-DPO pairwise preference loss
# rewards.py: class RewardModel(Protocol): score(media, prompts) -> Tensor
#             registry + composite weighted rewards
```

### `avgen.eval`
```python
class Metric(Protocol): update(...) ; compute() -> dict[str,float] ; reset()
register_metric(name) / build_metric(name, **kw) / list_metrics()
# built-ins with no optional deps: temporal consistency, motion magnitude,
# flicker, AV sync proxy, first-frame fidelity, seam continuity
# gated backends (FVD, CLIP, etc.) declared but raise a clear error if missing
class EvalReport  # JSON-safe, comparable across runs, records pinned config
```

### `avgen.config`
```python
# dataclass + YAML + dotted CLI overrides. NO hydra, NO omegaconf, NO pydantic.
@dataclass RunConfig: model, data, train, parallel, checkpoint, telemetry, eval
load_config(path, *, overrides: Sequence[str] = ()) -> RunConfig
# override syntax: "train.lr=1e-4" "parallel.context=8"
save_config(config, path) ; config_diff(a, b) -> dict
```

### `avgen.cli`
```
avgen train      --config configs/... [overrides]
avgen generate   --checkpoint ... --prompt ...
avgen simulate   --config ... --world-size 1024   # prints the plan table
avgen plan       --model 2b --world-size 512 --seq-len 65536
avgen eval       --checkpoint ... --data ...
avgen checkpoint {convert,inspect,export}
avgen data       {ingest,shard,inspect}
```

---

## 5. File ownership — do not write outside your set

| Owner | Paths |
|---|---|
| lead (me) | `src/avgen/core/**`, `src/avgen/parallel/**`, `src/avgen/simulate/**`, `src/avgen/__init__.py`, `README.md`, `CONTRACTS.md` |
| models | `src/avgen/models/**` |
| train | `src/avgen/train/**` |
| data | `src/avgen/data/**` |
| checkpoint+telemetry | `src/avgen/checkpoint/**`, `src/avgen/telemetry/**` |
| infer+codecs | `src/avgen/infer/**`, `src/avgen/codecs/**` |
| finetune+rl | `src/avgen/finetune/**`, `src/avgen/rl/**` |
| eval+config+cli | `src/avgen/eval/**`, `src/avgen/config/**`, `src/avgen/cli/**`, `configs/**` |
| docs+ci | `docs/**`, `.github/**`, `Makefile`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `GOVERNANCE.md`, `CHANGELOG.md`, `CITATION.cff`, `NOTICE`, `.pre-commit-config.yaml`, `.gitignore` |
| tests | `tests/**`, `examples/**`, `benchmarks/**` |

---

## 6. Rules that exist because of scale

Violating these produces bugs that only appear on a real cluster.

1. **RNG varies on `data_rank` only.** CP/TP/PP ranks holding shards of one sample draw identical noise. Use `RNGStreams.for_rank(seed, data_rank=...)`.
2. **Data shards on `data_rank` only.** Same reason.
3. **Reduce losses/metrics over the `dp_cp` mesh**, never the whole world. Use `avgen.parallel.data_mesh(mesh)`.
4. **Never call `.item()` inside a training step.** Accumulate on device; sync once at log cadence.
5. **`sample_ids` stay on CPU.**
6. **Gradient clipping goes through `avgen.parallel.clip_grad_norm`** — it handles DTensor and pipeline stages.
7. **Optional deps import lazily**, inside the function, with a `RuntimeError` naming the extra.
8. **Validation at boundaries, never in the hot loop.** `validate()` is called by the loader and the sampler, not per block.
9. **A model exposes `.blocks: nn.ModuleList`** or FSDP/AC/compile cannot wrap per block.
10. **Anything checkpointable implements `Stateful`** so DCP can save it without knowing what it is.
11. **No `torch.cuda` call at import time.** The package must import on a CPU-only machine.
12. **Determinism:** the same seed, config, and rank count reproduce the same loss curve. No `set` iteration order in anything that affects computation.

---

## 7. Definition of done for your subsystem

- Every public symbol exported from the package `__init__.py` with a sorted `__all__`.
- `ruff check` and `ruff format --check` clean.
- `mypy --strict` clean for your files.
- Docstrings complete, comments explain *why*.
- Your subsystem imports cleanly on CPU with only `torch`, `numpy`, `pyyaml`, `safetensors` installed.
- You wrote at least a smoke path that runs on CPU with tiny shapes.
- Report back: what you built, any contract you needed that was missing, anything you could not finish.
