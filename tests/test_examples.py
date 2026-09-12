"""Execute every shipped example and benchmark.

An example that no longer runs is worse than no example: it is the first code a
new user copies, and it fails on their machine rather than in CI. These are
complete programs, not fragments, so running them is the honest check — an
import test would pass on all five even if every signature had drifted.

They are also the framework's only end-to-end coverage of the paths a *reader*
takes rather than the paths the CLI takes, which is why a broken signature here
has repeatedly meant a broken public API rather than a broken example.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((REPO_ROOT / "examples").glob("*.py"))

# Each of these spawns a fresh interpreter that imports torch, so they cost
# seconds apiece regardless of how little work the example itself does. They are
# a release gate rather than a pre-push one: `make test` runs them, the fast
# subset does not.
pytestmark = pytest.mark.slow


def _run(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
    )


def test_the_examples_directory_is_not_empty() -> None:
    """Guards the glob: a typo in the path would make every test below vacuous."""
    assert len(EXAMPLES) >= 5


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_the_example_runs(script: Path) -> None:
    result = _run(script)
    assert result.returncode == 0, (
        f"{script.name} failed\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), f"{script.name} printed nothing"


def test_the_benchmark_runs() -> None:
    result = _run(
        REPO_ROOT / "benchmarks" / "bench_step.py", "--steps", "2", "--warmup", "1"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "tokens/s" in result.stdout


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_the_example_is_listed_in_the_index(script: Path) -> None:
    """An example nobody is pointed at is an example nobody reads."""
    index = (REPO_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    if script.name == "README.md":  # pragma: no cover - defensive
        return
    assert script.name in index, f"{script.name} is missing from examples/README.md"
