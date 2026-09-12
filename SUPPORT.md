# Getting help

Four channels, and the right one depends on what you have.

## I have a question about using avgen

**[Discussions → Q&A](https://github.com/thesvj/avgen/discussions/categories/q-a).**

Before asking, the answer is often in one of these:

| Question | Page |
|---|---|
| How do I install it, and which extras do I need? | [Installation](https://thesvj.github.io/avgen/getting-started/installation/) |
| What does a token stream look like? What is this framework doing? | [Quickstart](https://thesvj.github.io/avgen/getting-started/quickstart/) |
| How do I actually start a training run? | [Your first training run](https://thesvj.github.io/avgen/getting-started/first-training-run/) |
| Which parallelism do I want for my model and my cluster? | [Parallelism](https://thesvj.github.io/avgen/guides/parallelism/) · [Scaling to 1000 GPUs](https://thesvj.github.io/avgen/guides/scaling-to-1000-gpus/) |
| It OOMs. | [Memory](https://thesvj.github.io/avgen/guides/memory/) |
| It hangs, crashes, or the loss is wrong. | [Troubleshooting](https://thesvj.github.io/avgen/guides/troubleshooting/) |
| Will this configuration fit before I book the nodes? | `avgen plan --help`, and [Simulation](https://thesvj.github.io/avgen/guides/simulation/) |

The [`examples/`](examples/) directory has five complete programs that run on a
CPU in seconds, and every config in [`configs/`](configs/) is annotated with why
each choice was made.

## I think I found a bug

**[Open a bug report](https://github.com/thesvj/avgen/issues/new?template=bug_report.yml).**

The form asks for the avgen version, the torch version, the GPU, the world size,
the parallelism degrees and the exact config. All six, because a distributed
training bug is not reproducible without them — a report missing the parallelism
degrees is usually unactionable, and we would rather ask once in a form than
three times in comments.

**Before filing**, two things resolve a large share of reports on their own:

```bash
avgen info                    # what is installed, importable, and visible
avgen simulate --config your_config.yaml --world-size <N>
```

`avgen info` catches a broken install or an invisible GPU. `avgen simulate`
catches a configuration that was never going to fit, which reads at runtime as
an OOM of mysterious origin.

## I want a feature, or I want to change a contract

A feature request is an [issue](https://github.com/thesvj/avgen/issues/new?template=feature_request.yml).

A change to a **frozen contract** — anything in `CONTRACTS.md` §3 or §4, the
mesh dimension order, the model ABI, the checkpoint layout — goes through the
[RFC process](https://github.com/thesvj/avgen/issues/new?template=rfc.yml)
instead, because every subsystem imports those and a unilateral change breaks
all of them at once. [`GOVERNANCE.md`](GOVERNANCE.md) describes how a decision
gets made.

## I found a security vulnerability

**Do not open an issue.** See [`SECURITY.md`](SECURITY.md) for private
disclosure.

---

## What this project does not promise

Stating it plainly is more useful than leaving you to find out:

- **No support for running a competitor's checkpoint.** avgen trains models; it
  does not claim to load arbitrary third-party weights.
- **No help tuning your dataset.** The framework ships no dataset and no
  dataset-specific thresholds, deliberately.
- **The simulator estimates time, and says so.** Its shape, sharding, memory and
  collective accounting is exact; its step-time prediction is a model. A
  disagreement with a real run larger than roughly 20% is a bug worth reporting
  — a disagreement of 5% is the tool working as documented.
- **No commercial support.** This is an Apache-2.0 project maintained in public.

## Response times

Best effort, from a small maintainer group. A security report is acknowledged in
three business days ([`SECURITY.md`](SECURITY.md)); nothing else carries an SLA.
A well-formed bug report with a reproduction gets attention fastest — not as a
reward, but because it is the only kind that can be acted on without a round
trip.
