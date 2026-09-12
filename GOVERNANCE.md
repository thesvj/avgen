# avgen governance

avgen is an open-source project under Apache-2.0. This document describes who
decides what, how those people are chosen, and how a change that breaks users
gets made.

It is deliberately lightweight. The project is young; the process should be the
smallest one that keeps the codebase coherent.

## Current state, stated plainly

**Today there is one maintainer.** The roles, votes and comment periods below
describe how the project is meant to operate as people join, not a committee
that already exists — and a document that pretended otherwise would waste the
time of the first person who tried to use it.

What that means in practice right now:

- Everything in the **Decision process** section still applies to *you*: a
  change is discussed in the open, in the issue or the pull request, and the
  reasoning is written down. That part is not contingent on headcount.
- The comment periods are real, not theatre. An RFC still gets its ten working
  days, because the point of the delay is to give anyone reading time to object,
  and there is no quorum to shortcut.
- Where this document says "maintainer consensus" or "steering-group majority",
  read it today as one person's decision, recorded with its reasoning in the
  issue so it can be argued with later.
- `.github/CODEOWNERS` currently routes every path to the same person. It is
  split by area so the split is visible and ready, not because the areas have
  separate owners yet.

The first people to become maintainers under the process below are the ones who
make it real. Until then, this is a commitment about how decisions will be made,
published in advance so it cannot be quietly redefined later to suit an outcome.

---

## Principles

1. **Technical decisions are made in public, in the issue or pull request.**
   A decision reached on a call is not a decision until it is written down in
   the tracker.
2. **The contracts are the product.** `CONTRACTS.md` and the frozen core APIs
   are what let eight people work on eight subsystems at once. Changing them is
   a governance act, not a refactor.
3. **Disagreement is resolved by evidence.** For a performance claim, that means
   a simulator table or a measured run, not an argument about which approach
   ought to be faster.
4. **Consensus first, votes as a fallback.** Most decisions are made by nobody
   objecting.

---

## Roles

### Contributor

Anyone who opens an issue, reviews a pull request, improves the docs, or lands
a patch. No formal status; no permissions beyond a fork.

### Reviewer

A contributor with sustained, high-quality involvement in one area. Reviewers
have no merge rights but their review is a required signal on pull requests in
their area, and they are listed in `.github/CODEOWNERS` for it.

Reviewers are nominated by a maintainer and confirmed by lazy consensus of the
maintainers (see below).

### Maintainer

Maintainers have write access to the repository. A maintainer:

- reviews and merges pull requests in their area,
- is responsible for the health of the area they own in `.github/CODEOWNERS`,
- triages issues in that area,
- votes on RFCs and on new maintainers.

Areas currently are: **core**, **parallel**, **simulate**, **models**,
**train**, **data**, **checkpoint & telemetry**, **infer & codecs**,
**finetune & rl**, **eval, config & cli**, **docs & CI**.

### Steering group

The maintainers of `core`, `parallel`, and `simulate`, plus the project lead.
The steering group exists for exactly two purposes: breaking the rare tie that
maintainer consensus cannot, and approving changes to the frozen contracts. It
does not review ordinary pull requests.

---

## Decision process

### Ordinary changes

A pull request may be merged when:

- CI is green,
- at least one maintainer of the affected area has approved it,
- no maintainer has an unresolved objection.

The author may not merge their own pull request unless it is a docs typo or a
CI-only fix, and even then a second pair of eyes is preferred.

### Lazy consensus

For proposals that are not code — adding a reviewer, adopting a policy,
scheduling a release — a maintainer posts the proposal in an issue and, if no
maintainer objects within **five working days**, it is adopted.

Objections must be technical and must name what would resolve them. "I don't
like it" is not an objection; "this makes the mesh order depend on the launcher,
which breaks the checkpoint resharding invariant" is.

### Escalation

If maintainers cannot converge within two weeks, any maintainer may escalate to
the steering group, which decides by simple majority within one week. The
decision and its reasoning are posted in the original issue.

---

## Becoming a maintainer

There is no application form and no fixed contribution count. The bar is
**demonstrated judgement in an area, over time**.

In practice a nomination is credible when the person has, over roughly three
months:

- landed several non-trivial changes in one area,
- reviewed other people's changes in that area usefully — meaning they caught
  something,
- shown they understand the scale rules in `CONTRACTS.md` §6, which is the
  thing that distinguishes a reviewer who helps from one who rubber-stamps,
- responded to issues in their area.

Process:

1. An existing maintainer opens a private issue nominating the candidate, with
   links to the work.
2. Maintainers discuss for five working days.
3. Approval requires a majority of maintainers and no objection from the
   steering group.
4. The new maintainer is added to `.github/CODEOWNERS` and given write access
   in the same pull request.

### Stepping down and inactivity

Maintainers who have not participated for **six months** are moved to emeritus
status, keeping the credit and losing the write bit. This is not a judgement;
it is a security practice. Coming back is a one-line request.

---

## RFC process for breaking changes

An RFC is required for any change that:

- alters a frozen contract in `CONTRACTS.md` §3 or §4,
- changes the meaning, name, or ordering of a mesh dimension,
- changes the on-disk format of a checkpoint, dataset shard, or config,
- removes or renames a public symbol,
- changes a default that silently changes training results — the timestep
  shift, the RNG stream layout, the loss reduction mesh, the default
  parallelism plan,
- adds a new **required** dependency to the core package.

Adding an optional extra, a new registry entry, or a new module does **not**
need an RFC.

### How it works

1. **Open an issue with the "RFC" template.** State the problem first. An RFC
   that opens with a solution usually gets the problem wrong.
2. The issue must contain:
   - **Problem** — what is broken or impossible today, with a concrete case.
   - **Proposal** — the new contract, written out as it would appear in
     `CONTRACTS.md`.
   - **Alternatives considered** — and why each lost. An RFC with no
     alternatives section has not been thought about yet.
   - **Compatibility impact** — what breaks, for whom, and what the migration
     is. For anything touching parallelism or checkpoints, include the
     simulator output before and after.
   - **Migration plan** — deprecation shim, version, and removal release.
3. **Comment period: ten working days**, extended if anyone asks.
4. **Decision:** maintainer consensus, or steering-group majority if consensus
   is not reached. The decision is recorded in the issue, with reasoning, and
   the issue is labelled `rfc-accepted` or `rfc-declined`.
5. **Implementation** is a separate pull request that links the accepted RFC and
   updates `CONTRACTS.md`, the docs, and `CHANGELOG.md` together.

### Deprecation policy

Before 1.0, a breaking change may ship in a minor release with:

- a `CHANGELOG.md` entry under **Removed** or **Changed** describing the
  migration,
- a `DeprecationWarning`-emitting shim kept for at least one minor release
  where a shim is technically possible.

After 1.0, breaking changes ship only in a major release.

---

## Releases

Releases are cut by any maintainer, announced in advance by lazy consensus, and
published to PyPI by `.github/workflows/release.yml` via trusted publishing
(OIDC). No individual holds a PyPI token.

A release requires:

- CI green on `main`, including the simulator workflow,
- the GPU nightly green on the release commit or the most recent nightly,
- `CHANGELOG.md` moved from `[Unreleased]` to the version with a date,
- a signed, annotated tag `v<version>`.

Versioning follows [Semantic Versioning](https://semver.org/), with the pre-1.0
caveat above: `0.x` minor releases may break.

---

## Amending this document

Changes to `GOVERNANCE.md` follow the RFC process, with a decision by
steering-group majority.
