# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While avgen is `0.x`, minor releases may contain breaking changes; each one is
listed under **Changed** or **Removed** with its migration.

## [Unreleased]

Nothing yet.

## [0.1.0] - Unreleased

First public release. The scope of this release is a complete, PyTorch-native
training stack for video and audio-video diffusion transformers, plus the
simulator that makes a large-scale parallelism plan reviewable before it is run.

### Added

#### Core representation

- `avgen.core.tokens` — `TokenStream`, the sequence-first representation used by
  every model interior, carrying per-token physical coordinates (seconds for
  time, latent pixels for space), a validity mask, per-sample or per-token noise
  level, and a clean-anchor mask that survives context-parallel sharding.
  `PatchLayout` holds the static geometry; `TextContext` holds cross-attention
  conditioning.
- `avgen.core.patchify` — `Patchifier` protocol and `GridPatchifier`, the single
  boundary where dense 5-D latent grids become sequences and back. Exact
  round-trip; time-major, then row, then column ordering.
- `avgen.core.model_input` — `ModelInput`/`ModelOutput`, the model ABI. Every
  model is `forward(ModelInput) -> ModelOutput`, in token space.
- `avgen.core.batch` — `MediaBatch`, `MediaBatchSpec` with exact rational
  timebases, `ConditionMode` covering joint AV, video-to-audio, audio-to-video,
  continuation, inpaint, image-to-video, and video-to-video.
- `avgen.core.rng` — `RNGStreams` with separated noise, timestep, conditioning,
  and sampler generators, seeded from `data_rank` only so context- and
  tensor-parallel ranks holding shards of one sample draw identical noise.
- `avgen.core.state` — `Stateful`, `TrainState`, and the `LRSchedule`, `EMA`,
  `DataCursor` protocols that Distributed Checkpoint saves without knowing what
  they are.
- `avgen.core.metrics` — `StepMetrics`, accumulated on device with exactly one
  host synchronisation per log interval.

#### Parallelism

- Five explicit, validated axes — `dp_replicate`, `dp_shard`, `cp`, `tp`, `pp` —
  in `ParallelDims`, with the mesh built in the order
  `(pp, dp_replicate, dp_shard, cp, tp)` so tensor-parallel ranks land inside one
  NVLink domain and pipeline ranks are furthest apart.
- FSDP2 (`fully_shard`) sharding with configurable prefetch depth, mixed
  precision policy, and CPU offload.
- Tensor parallelism with sequence parallelism, driven by a declarative plan
  (`standard_block_plan`, `standard_root_plan`) that a model opts into by
  implementing `TensorParallelizable`.
- Context parallelism for video: `shard_stream`, `gather_stream`,
  `pad_to_multiple`, and `context_parallel_region` over PyTorch's ring
  attention. This is the axis that decides whether a long clip fits at all.
- Pipeline parallelism with balanced splitting, interleaved schedules, and a
  bubble-fraction model.
- Selective-op, selective-layer, and full activation checkpointing.
- bfloat16 mixed precision with fp32 reductions, and optional fp8 conversion via
  torchao under the `quant` extra.
- Collectives that respect the mesh: `all_reduce_{sum,mean,max}`, `data_mesh`,
  and `clip_grad_norm` that handles DTensor and pipeline stages.

#### Simulator

- `avgen.simulate.world` — build a real 1024-rank `DeviceMesh`, apply the real
  parallel plans, and inspect the result on one CPU machine, using
  `FakeProcessGroup`, `FakeTensorMode`, and meta-device initialisation.
- `avgen.simulate.memory` — closed-form per-rank memory with a breakdown by
  parameters, gradients, optimizer state, activations, FSDP gather peak, and
  workspace, plus `measure_memory` to calibrate against a real run.
- `avgen.simulate.comms` — analytic cost of ring attention, FSDP all-gather and
  reduce-scatter, tensor-parallel collectives, and HSDP all-reduce, over
  calibratable interconnect tiers, reporting exposed communication and scaling
  efficiency.
- `avgen.simulate.compute` — transformer FLOP counting with the attention term
  separated out, roofline timing, MFU and HFU, and an activation-policy
  suggester.
- `avgen.simulate.plan` — enumerate every valid factorisation of a world size,
  price each one, and rank what fits. A 1024-GPU search takes under a second.
- `avgen.simulate.chakra` — PyTorch execution-trace capture and the conversion
  path to an MLCommons Chakra trace for ASTRA-sim network simulation.

#### Models, training, data

- `VideoDiT` and `AVDiT` diffusion transformers behind a model registry.
- Rectified-flow / flow-matching objective with a resolution-dependent timestep
  shift (`shifted_logit_normal`), plus uniform, logit-normal, and mode samplers.
- Conditioning sampler covering all `ConditionMode` values, with text dropout
  for classifier-free guidance.
- `ShardedEMA`, DTensor-safe, with bf16 storage.
- `constant`, `cosine`, `linear`, and warmup-stable-decay schedules.
- Resumable `DataSource` protocol, resolution/duration bucketing, a
  dependency-free `SyntheticSource` for tests and smoke runs, and mmap +
  safetensors shards with a sha256 manifest and atomic commit.

#### Checkpointing, inference, and the rest

- Distributed Checkpoint save/load that **reshards across a different rank
  count**, with async save, retention policy, and resume-latest.
- `export_safetensors` and `export_huggingface` for release artifacts.
- Samplers `euler`, `heun`, `dpmpp_2m`, `res_multistep`; classifier-free
  guidance with rescale, modality-specific guidance, and APG.
- LoRA and DoRA adapters, layer freezing, and ControlNet-style control adapters.
- Flow-GRPO and diffusion-DPO post-training with a reward-model registry.
- Evaluation metrics that need no optional dependency — temporal consistency,
  motion magnitude, flicker, an AV-sync proxy, first-frame fidelity, seam
  continuity — and gated backends that fail with a clear message.
- Dataclass + YAML configuration with dotted CLI overrides. No Hydra, no
  OmegaConf, no Pydantic.
- `avgen` CLI: `train`, `generate`, `simulate`, `plan`, `eval`, `checkpoint`,
  `data`.

#### Project

- Documentation site built with MkDocs Material, with an auto-generated API
  reference. The build runs in strict mode, so a docstring documenting a
  parameter that does not exist fails CI.
- CI on Python 3.11, 3.12 and 3.13; a simulator workflow that fails the build
  when a reference parallelism plan stops fitting or its predicted MFU
  regresses; a nightly GPU job on a self-hosted runner.
- `examples/` — five complete programs that run on a CPU in seconds, each
  executed by the test suite, because an example that no longer runs is the
  first code a new user copies.
- `benchmarks/bench_step.py` — measures a real training step and prints the
  `avgen plan` invocation to compare it against, so the simulator's error stays
  a known quantity rather than an assumption.
- `CONTRIBUTING.md` and `CONTRACTS.md`, the latter being the interface
  agreement between subsystems.
- Config errors name the fix, not just the mistake: a typo, a field another
  framework spells differently, a field that lives in another section (named by
  its full path), and a setting that comes from the launcher rather than the
  config all produce a specific suggestion instead of a list of valid keys.

### Notes

- Requires Python ≥ 3.11 and `torch >= 2.6`.
- The core package depends only on torch, numpy, pyyaml, and safetensors.
  Everything else is an opt-in extra: `text`, `codecs`, `data`, `tracking`,
  `quant`, `rewards`, `fault-tolerance`.
- Distributed execution is entirely PyTorch-native, and launching is `torchrun`.
  Ray, DeepSpeed, Megatron-Core, Accelerate and Lightning are not dependencies
  in any form, optional or otherwise.
- The `hps_v2` reward is registered but `hpsv2` is deliberately not declared as
  an extra: it pins `pytest==7.2.0`, which makes a universal lock resolution
  unsatisfiable. Install it into its own environment; the reward raises with
  that instruction.

[Unreleased]: https://github.com/thesvj/avgen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thesvj/avgen/releases/tag/v0.1.0
