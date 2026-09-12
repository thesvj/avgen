# avgen configuration

Plain YAML over plain frozen dataclasses. The only dependency is `pyyaml`.

There is no hydra, no omegaconf, and no pydantic here, and that is a decision
rather than an omission. A training config is the most-read artifact in a
research project — it gets pasted into issues, diffed across runs, attached to
papers, and read by people who will never install the framework. Making it
require a library to understand costs more than the library saves. Composition,
interpolation, and overrides are implemented in ~400 lines in
`src/avgen/config/`, and every key in every file below is a dataclass field name
you can look up in `src/avgen/config/schema.py`.

---

## Layout

```
configs/
├── model/      architecture only          tiny, dit_300m, dit_2b, dit_5b, dit_14b, avdit_2b
├── data/       input pipeline             synthetic, latent_shards_template
├── train/      complete runs by scale     smoke_cpu, single_gpu, node_8gpu,
│                                          multinode_64, multinode_1024
├── finetune/   adaptation                 lora, full, control
├── rl/         post-training              grpo, dpo
└── eval/       measurement                default
```

Files under `model/`, `data/`, and `eval/` are **fragments** — they set one
section and are meant to be composed. Files under `train/`, `finetune/`, and
`rl/` are **complete runs** that compose the fragments they need.

---

## Composition with `_base_`

A file may name one parent or several. Parents are merged in order, then the
file itself is merged on top, so later always wins.

```yaml
_base_:
  - ../model/dit_2b.yaml
  - ../data/latent_shards_template.yaml

train:
  lr: 8.0e-5        # overrides whatever the parents set
```

Paths are relative **to the file that names them**, so a config tree can be
copied or vendored without rewriting every path inside it.

Merging is deep for mappings and **replacing** for lists. That asymmetry is
deliberate: merging a list of data buckets element-by-element looks helpful
right until someone prepends a bucket, every downstream index shifts by one, and
the config quietly means something nobody intended. If you want to change one
bucket, restate the list.

Cycles are detected and reported with the full chain.

---

## Environment interpolation

```yaml
data:
  root: ${env:AVGEN_DATA_ROOT}                    # required; errors if unset
  manifest: ${env:AVGEN_MANIFEST:manifest.json}   # with a default
```

That is the whole interpolation language. There is no arithmetic, no
cross-reference, and no expression evaluation, because a config file that can
execute is a config file you have to audit.

An unset variable with no default is a hard error naming the variable. It never
silently becomes an empty string, because an empty `data.root` produces a run
that reads zero samples and reports a suspiciously smooth loss curve.

---

## Command-line overrides

Any config value can be set on the command line, after `--config`:

```bash
avgen train --config configs/train/node_8gpu.yaml \
    train.lr=1e-4 \
    parallel.context=8 \
    telemetry.loggers='[console, jsonl, tensorboard]' \
    data.buckets[0].height=512
```

Three properties worth knowing:

**Types come from the schema, not from the string.** `train.lr=1e-4` becomes the
float `0.0001` because `TrainConfig.lr` is annotated `float`. This is not a
detail: YAML 1.1 parses `1e-4` as the *string* `"1e-4"` because it lacks a
decimal point, so a loader that guesses from the text hands a string to your
optimizer. Reading the annotation is what makes the obvious spelling work.

**An unknown key is a hard error**, with the nearest valid field name suggested.
The suggestion covers typos and the names other trainers use for the same field,
so arriving from another framework costs one error rather than a search:

```
$ avgen train --config ... train.learning_rate=1e-4
avgen: error: unknown configuration key(s) in train: train.learning_rate;
did you mean 'lr'?
```

It also catches the two cases a per-section suggestion cannot. A field that
lives in a **different section** is named by its full path, because listing the
valid keys of the section you are looking in does not help when the field is
somewhere else:

```
$ avgen train --config ... train.gradient_checkpointing=true
avgen: error: unknown configuration key(s) in train: train.gradient_checkpointing;
did you mean 'parallel.activation.mode'? (it is in another section)
```

And a setting that is **not configuration at all** says where it really comes
from:

```
$ avgen train --config ... train.world_size=8
avgen: error: unknown configuration key(s) in train: train.world_size — the world
size comes from the launcher (torchrun sets WORLD_SIZE), not from the config; set
the parallelism degrees and avgen derives the rest
```

A silently ignored typo in a config is a wasted cluster run. `lr_warmup_steps`
instead of `warmup_steps` produces a job that trains happily on the wrong
schedule and nothing anywhere says so.

**List elements are addressable** with `[i]`, but only when the list exists in
the file. An entry conjured from the command line would have no siblings and no
defaults you chose, so avgen refuses it and says which file to edit.

---

## Every value is validated before anything is allocated

Validation lives in each dataclass's `__post_init__` and every message names the
field and echoes the rejected value:

```
avgen: error: invalid configuration in model: model.num_heads must divide
model.width; got width=2560 num_heads=24 (remainder 8)
```

Cross-section constraints are checked too. An `av_dit` model with
`data.audio_frames: 0` is rejected, because the audio tower would train on
empty tensors while the video loss looked perfectly healthy.

---

## The resolved config travels with the run

`checkpoint.save_config` (on by default) writes the **fully resolved** config —
after `_base_` composition, after environment interpolation, after every
command-line override — into the run's output directory, next to the
checkpoints. That file is the whole truth about the run, with nothing left to
reconstruct from shell history.

`avgen generate` and `avgen eval` look for it beside the checkpoint
automatically, so reproducing a sample does not require remembering which config
produced the weights.

To diff two runs:

```python
from avgen.config import config_diff, format_diff, load_config

a = load_config("runs/exp-a/config.yaml")
b = load_config("runs/exp-b/config.yaml")
print(format_diff(config_diff(a, b), left_name="a", right_name="b"))
```

---

## Choosing a `train/` config

| File | Scale | Parallelism | Why |
|---|---|---|---|
| `smoke_cpu` | 1 CPU | none | Full pipeline in seconds. No GPU, no dataset, no download. |
| `single_gpu` | 1 GPU | none | 300M at 16k tokens fits; nothing needs sharding. |
| `node_8gpu` | 8 GPU | `dp_shard=8` | 2B needs ~32 GB of weights+grads+AdamW state before activations; FSDP divides all three. |
| `multinode_64` | 64 GPU | `cp=4, dp_shard=16` | 65k tokens is the constraint. CP is the only axis that reduces per-rank sequence length. |
| `multinode_1024` | 1024 GPU | `cp=8, tp=8, hsdp 4x4` | Every axis doing a distinct job; TP pinned inside the NVLink domain. |

Each file explains its parallelism choice in comments at the top.

**Do not guess the next one up.** Two commands answer it in under a second each,
on a laptop, with no GPU:

```bash
# rank every valid factorisation for a model shape and world size
avgen plan --world-size 512 --seq-len 65536 --params 2e9 --depth 32 --width 2560

# price YOUR config, and check the mesh really applies at that scale
avgen simulate --config configs/train/multinode_64.yaml --world-size 512 --topology
```

---

## Note on the data configs

`configs/data/latent_shards_template.yaml` is a **template**. avgen ships no
dataset and names no source: no dataset names, no URLs, no quotas, no tuned
mixture weights or filter thresholds. The paths are blank, the buckets are
neutral placeholders, and the numbers that are set (`caption_dropout: 0.1`,
`drop_last: true`) are set because they are correct in general, not because they
were tuned on anything.

Latent dimensions in a bucket are **post-compression**. With a VAE at 8× spatial
and 4× temporal compression, a 480p 5-second clip at 24 fps is roughly
`frames: 30, height: 60, width: 106`. Substitute your own codec's ratios.

Before a long run:

```bash
avgen data validate /path/to/shards --config configs/train/node_8gpu.yaml
```

A shard set that is a fraction of a percent corrupt trains without complaint and
produces a model that is slightly worse for no visible reason.
