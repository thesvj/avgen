# Security

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through
[GitHub Security Advisories](https://github.com/thesvj/avgen/security/advisories/new),
or email **saij@iiitd.ac.in** if that does not work for you. Do include the
affected version, how to reproduce it, and the impact you think it has.

This is a small project, so there is no SLA to promise. What I will do is
acknowledge your report within a few days, tell you honestly whether I think it
is valid, and credit you in the advisory unless you would rather I did not.

## What is in scope

avgen is a training framework. It reads configs you write, checkpoints you
produce and data you point it at, and it runs on hardware you control. Most of
what it touches is already trusted input.

The parts worth reporting:

- anything that executes code from a config file, a dataset shard or a
  checkpoint that was not meant to be executable
- a path traversal or arbitrary write through a config field, a shard manifest
  or a checkpoint path
- a dependency we pull in with a known advisory against it

Loading an untrusted checkpoint is **not** in scope, and no framework can make
it safe. `torch.load` on a pickle you did not create runs whatever that pickle
says to run. avgen writes and reads its own weights through safetensors and
Distributed Checkpoint for exactly this reason, but if you point it at a
third-party `.pt` file, you are trusting whoever made it.

## Supported versions

avgen is pre-1.0. Fixes land on `main` and go out in the next patch release.
Older minor series are not backported.
