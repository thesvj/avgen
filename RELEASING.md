# Releasing avgen

For maintainers. Cutting a release is five commands; the reason it is written
down is that three of the five gates exist because of a specific way a release
goes wrong, and the notes below say which.

## Versioning

Semantic versioning, applied to the **public API and the on-disk formats** —
which for a training framework is the part that matters:

| Change | Bump |
|---|---|
| A frozen contract in `CONTRACTS.md` §3/§4 changes shape | major |
| The checkpoint layout stops loading older checkpoints | major |
| The mesh dimension order changes | major |
| A public symbol is removed or its signature narrows | major |
| New subsystem, model, sampler, metric, parallelism plan | minor |
| A default that changes a loss curve | minor, and say so loudly in the changelog |
| Bug fix, docs, performance, a new optional extra | patch |

Pre-1.0, the frozen contracts are the thing under version discipline. Everything
else may move in a minor release, and the changelog is where that is disclosed.

**A default that changes numerics is never a patch.** Someone resumes a run
across the upgrade and gets a different loss curve with nothing to point at.

## Cutting a release

```bash
# 1. Everything green, including the slow suites the fast subset skips.
make lint && make type && make test && make simulate

# 2. The reference plans still fit and hold their throughput floors.
#    Already covered by `make simulate`; read the table, do not just trust the
#    exit code. A 20% MFU regression at 1024 ranks exits zero.

# 3. Move the Unreleased section to a version, and date it.
$EDITOR CHANGELOG.md

# 4. Set the version. The tag must match this exactly or CI refuses the build.
$EDITOR pyproject.toml          # project.version
$EDITOR CITATION.cff            # version and date-released

# 5. Commit, tag, push.
git commit -s -am "release: v0.2.0"
git tag -s v0.2.0 -m "v0.2.0"
git push origin main v0.2.0
```

Pushing the tag is what publishes. `.github/workflows/release.yml` then:

1. **Refuses a tag that disagrees with `pyproject.toml`.** A release whose tag
   and metadata disagree cannot be reproduced from the source tree, and the
   discrepancy is invisible once it is on PyPI.
2. **Refuses a version with no `## [x.y.z]` section in `CHANGELOG.md`.** The
   changelog is the release notes; a release without one is a version number.
3. Builds the sdist and wheel and runs `twine check`.
4. Publishes to PyPI via **trusted publishing** — PyPI mints a short-lived token
   from the workflow's OIDC identity, so no API token exists in a secret, in a
   password manager, or in a leaked log. Configure the publisher once, at
   `https://pypi.org/manage/project/avgen/settings/publishing/`, for repository
   `thesvj/avgen`, workflow `release.yml`, environment `pypi`.
5. Attaches the artifacts to the GitHub release.

## Dry run

Before a first release, or after touching the workflow, run it without
publishing:

```bash
gh workflow run release.yml -f dry_run=true
```

That exercises the version check, the changelog check, the build and
`twine check`, and stops before PyPI.

To check the artifact locally, install the wheel into a clean environment with
nothing but the core dependencies and confirm the CLI works — this is what
catches a packaging error that every test in the repo passes through:

```bash
uv build --out-dir dist
uv venv /tmp/avgen-check --python 3.12
uv pip install --python /tmp/avgen-check/bin/python dist/*.whl
/tmp/avgen-check/bin/avgen info
/tmp/avgen-check/bin/avgen plan --world-size 8 --seq-len 4096 \
    --params 3e8 --depth 16 --width 1024
```

## After the release

- The docs site deploys from `main` on its own (`.github/workflows/docs.yml`).
  Confirm the new version's pages are live.
- Open a new `## [Unreleased]` section in `CHANGELOG.md`.
- If the release changed a frozen contract, the RFC that authorised it
  ([`GOVERNANCE.md`](GOVERNANCE.md)) gets a comment linking the release, so the
  decision and its landing are connected for whoever reads it next.

## If a release is broken

**Do not delete it from PyPI.** A version that once existed and then vanished
breaks every lockfile that pinned it, and the failure surfaces in other people's
CI with no explanation.

Yank instead — the version stays resolvable for anyone who already pinned it,
and stops being selected for anyone who has not. Yanking is a PyPI-side action
with no CLI: `twine` has only `upload`, `check` and `register`. Do it in the web
UI, under *Manage project → Releases → Options → Yank*:

```
https://pypi.org/manage/project/avgen/releases/
```

Then fix forward with a patch release, and note the yank in the changelog under
the broken version, saying what was wrong with it. A yanked version with no
explanation sends everyone who pinned it to the issue tracker.
