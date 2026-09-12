"""Drive the real-collective checks from pytest.

The checks themselves live in ``tests/distributed/run_gloo_checks.py`` and run
under ``torchrun``, because every rank must execute them in the same order —
pytest's collection and fixtures do not guarantee that, and a rank that skips a
check its peers are running deadlocks the job on the next collective.

This module spawns that script as a subprocess so the properties it verifies are
part of the ordinary suite rather than something a person has to remember to
run. It uses **gloo on CPU**: not NCCL, not a cluster, but enough to make the
collectives move real bytes, which is the whole point. Under a fake process
group every one of these assertions passes vacuously.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tests" / "distributed" / "run_gloo_checks.py"

# Includes 3 deliberately: a prime, non-power-of-two world size is where
# assumptions about even division surface, and it is the case a framework tested
# only on 2/4/8 will have silently broken.
RANK_COUNTS = [2, 3, 4]


def _run(nproc: int) -> subprocess.CompletedProcess[str]:
    """Run the check script under torchrun with the given rank count."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            str(SCRIPT),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
    )


@pytest.mark.slow
@pytest.mark.parametrize("nproc", RANK_COUNTS)
def test_real_collectives_hold_at_every_rank_count(nproc: int) -> None:
    """Every distributed property must hold with collectives that move bytes.

    What this covers that nothing else can: that a reduction computes the actual
    mean, that the gathered gradient equals the average of the per-rank
    gradients, that replicas stay identical over many steps on different data,
    that the context-parallel loss identity holds, that a sharded checkpoint
    round-trips, that a NaN on one rank neither desynchronises nor deadlocks the
    others, and that data shards partition the corpus.
    """
    if torch.get_num_threads() < 1:  # pragma: no cover - defensive
        pytest.skip("no CPU threads available")
    result = _run(nproc)
    passed = result.stdout.count("  ok    ")
    failed = result.stdout.count("  FAIL  ")
    assert result.returncode == 0 and failed == 0, (
        f"{failed} check(s) failed at {nproc} ranks\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-1500:]}"
    )
    assert passed >= 7, f"expected at least 7 checks to run, saw {passed}"
