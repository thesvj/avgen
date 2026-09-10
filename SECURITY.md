# Security Policy

## Supported versions

avgen is pre-1.0. Security fixes land on `main` and are released in the next
patch version of the current minor series. Older minor series are not
backported.

| Version | Supported |
|---|---|
| `0.1.x` | Yes — current series |
| `< 0.1` | No |

Once avgen reaches 1.0 this table becomes "current minor plus the previous
minor", and this section will be updated in the same release.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report privately through GitHub Security Advisories:

1. Go to https://github.com/avgen-project/avgen/security/advisories/new
2. Describe the issue, the affected version, and how to reproduce it.
3. Include the impact you believe it has, and any suggested fix.

If you cannot use GitHub Security Advisories, email
**security@avgen-project.org** *(placeholder — replace with the real contact
before the first public release)*.

### What to expect

| Stage | Target |
|---|---|
| Acknowledgement of your report | 3 business days |
| Initial assessment (valid / not / severity) | 10 business days |
| Fix or documented mitigation for a confirmed high-severity issue | 90 days |

We will keep you updated as the assessment progresses, credit you in the
advisory unless you ask us not to, and coordinate disclosure timing with you.

## Scope

### In scope

- Code execution, path traversal, or privilege escalation triggered by loading
  an avgen **config file**, **checkpoint**, or **dataset shard**.
- Unsafe deserialisation in `avgen.checkpoint`, `avgen.config`, or
  `avgen.data.shard`.
- Command injection or unsafe subprocess use in the CLI or launchers.
- Secrets (API keys, tokens) leaked into logs, checkpoints, or telemetry
  payloads.
- Dependency vulnerabilities that avgen's own code makes exploitable.

### Out of scope

- **Model weights and their behaviour.** avgen is a training framework. What a
  model trained with it generates, memorises, or refuses to generate is not a
  vulnerability in avgen. Report model-behaviour concerns to whoever published
  the weights.
- **Training data.** avgen ships no datasets. The contents, licensing, and
  provenance of the data you train on are yours.
- **Untrusted checkpoints from third parties.** `torch.load` on an arbitrary
  pickle is remote code execution by design; avgen defaults to safetensors and
  Distributed Checkpoint for exactly this reason. Loading a hostile pickle you
  chose to trust is not an avgen vulnerability. If you find a path where avgen
  reaches `torch.load` with `weights_only=False` on a file the user did not
  explicitly opt into, **that is in scope** — please report it.
- Vulnerabilities in PyTorch, CUDA, NCCL, or other upstream dependencies.
  Report those upstream; tell us too if avgen needs a version pin.
- Denial of service through resource exhaustion in a training job you launched
  yourself (OOM, filling a disk, saturating a fabric).
- Findings from automated scanners with no demonstrated exploit path.

## Hardening notes

- Checkpoints are written with PyTorch Distributed Checkpoint and exported with
  `safetensors`. Neither format executes code on load.
- Configs are parsed with `yaml.safe_load`. `avgen.config` does not use
  `yaml.load`, `eval`, or arbitrary Python in configs.
- Optional dependencies are imported lazily, which keeps the attack surface of
  a minimal cluster image to torch, numpy, pyyaml, and safetensors.
