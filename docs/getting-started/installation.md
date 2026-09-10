# Installation

## Requirements

| | |
|---|---|
| Python | ≥ 3.11 |
| PyTorch | ≥ 2.6 |
| OS | Linux. macOS works for CPU development and the simulator; nothing distributed is tested there. |
| GPU | Not required to install, develop, or run the simulator. Required to train. |

PyTorch 2.6 is a floor, not a preference. Below it, `fully_shard`, the DTensor
APIs the tensor-parallel layer uses, and the context-parallel entry points are
either absent or differently shaped.

## Install

=== "uv (recommended)"

    ```bash
    uv add avgen
    ```

=== "pip"

    ```bash
    pip install avgen
    ```

=== "from source"

    ```bash
    git clone https://github.com/avgen-project/avgen
    cd avgen
    uv sync --all-extras --group dev --group docs
    ```

## What the core install gives you

`torch`, `numpy`, `pyyaml`, `safetensors`. That is the entire dependency set,
and it is deliberate: a cluster image that pulls a transitive graph of a hundred
packages onto 1024 ranks costs startup time on every job and gives you a hundred
more things that can break at scale.

With only the core installed you can:

- import every subpackage,
- build a model, train it on synthetic data, and checkpoint it,
- run the full parallelism simulator at any world size,
- run the whole non-GPU test suite.

What you cannot do is encode real video or embed real text — for that you need
a codec and a text tower, which are extras.

## Extras

| Extra | Pulls in | You need it when |
|---|---|---|
| `text` | `transformers`, `sentencepiece`, `accelerate` | Encoding prompts with a frozen T5/Gemma/Qwen tower. Usually done **offline**, once, into shards. |
| `codecs` | `diffusers`, `accelerate` | Encoding pixels to latents with a pretrained VAE, or decoding samples back. Also usually offline. |
| `data` | `pyarrow`, `av`, `torchaudio` | The offline ingest pipeline: decoding media files and reading columnar datasets. |
| `tracking` | `tensorboard` | The TensorBoard logger. avgen ships console, JSONL, and no-op loggers with no extra. |
| `quant` | `torchao` | fp8 training. |
| `fault-tolerance` | `torchft` | Per-step recovery without a full job restart. |
| `all` | `text`, `codecs`, `data`, `tracking`, `quant` | Convenience. Excludes `fault-tolerance` deliberately — it changes process-group semantics and should be an explicit choice. |

```bash
uv add 'avgen[text,codecs]'
```

!!! note "Extras are lazy"

    An optional dependency is never imported at module scope. It is imported
    inside the function that needs it, and if it is missing you get a
    `RuntimeError` naming the extra to install — not an `ImportError` five
    frames deep. This is enforced by a CI job that installs the core only and
    fails if any optional package appears in `sys.modules` after
    `import avgen`.

## Which extras a training job actually needs

Almost none, if you prepare data properly.

The recommended pipeline encodes video to latents and prompts to text features
**offline**, writing them into shards. The training job then reads tensors and
never touches a VAE or a text encoder. That means:

- **Ingest machine**: `avgen[data,codecs,text]`.
- **Training cluster**: core only, plus `tracking` if you want TensorBoard.

This is not just about image size. A VAE forward pass inside the training loop
is a serial bottleneck that scales with your data rate, competes for the same
GPU memory the model needs, and makes every run non-reproducible against a
different `diffusers` version.

## Verify

```bash
python -c "import avgen; print(avgen.__version__)"
```

Check that the parallelism layer works, with no GPU and no cluster:

```python
from avgen.parallel import ParallelDims

dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
print(dims.describe())
# world=1024 dp_shard=16 cp=8 tp=8
```

Then check the simulator, which is the real smoke test:

```bash
avgen plan --model 2b --world-size 512 --seq-len 72000
```

If that prints a ranked table of plans, everything that matters is installed.

## GPU notes

avgen makes **no `torch.cuda` call at import time**, so the package imports and
the simulator runs on a CPU-only machine. Verify your GPU install separately:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.nccl.version())"
```

For multi-node training you also need a working NCCL setup. The two settings
worth checking before your first multi-node run are `NCCL_IB_HCA` (the right
adapters) and `NCCL_SOCKET_IFNAME` (the right interface); see
[Troubleshooting](../guides/troubleshooting.md).

### CPU-only PyTorch for development

Installing the CUDA wheel to run tests on a laptop wastes about 2.5 GB:

```bash
UV_TORCH_BACKEND=cpu uv sync --group dev
```

CI does exactly this for every job except the nightly GPU run.

## Development install

```bash
git clone https://github.com/avgen-project/avgen
cd avgen
make dev        # every extra, dev + docs groups, pre-commit hooks
make test-fast  # should pass in well under a minute
```

See [`CONTRIBUTING.md`](https://github.com/avgen-project/avgen/blob/main/CONTRIBUTING.md)
for the full loop.
