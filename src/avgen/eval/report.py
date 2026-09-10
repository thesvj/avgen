"""Evaluation reports that refuse to be compared when they are not comparable.

**The problem this module exists to solve.** Most reported video-generation
comparisons are invalid, and the reason is boring: the two numbers were produced
under different sampling settings. Twenty steps against fifty. Guidance 5
against guidance 7.5. A different sampler. A different prompt set. A different
frame count. Each of those moves the headline metric by more than the
architectural difference being claimed, and none of them is usually stated. The
table looks like evidence and is not.

avgen's answer is mechanical rather than cultural. Every report carries an
:class:`EvalPin` — the exact sampling configuration, prompt-set identity,
resolution, and checkpoint the numbers came from — and
:func:`compare_reports` **raises** when two pins differ. You cannot accidentally
produce an invalid comparison with this API. You can produce a deliberate one,
by passing ``allow=`` for the specific field you are deliberately varying, and
that argument reads in a diff as exactly what it is.

The pin is deliberately strict about things people think are harmless:

* **The prompt set**, by content hash. Adding two prompts changes the number.
* **The metric implementation**, by name and configured parameters. A threshold
  changed in a metric constructor is a changed metric.
* **The sample count.** Fréchet-style estimators are biased in N.
* **The seed**, because for the small prompt sets people actually use, seed
  variance is comparable to the differences being reported. Two reports at
  different seeds *can* be compared with ``allow=("seed",)``, and that is the
  honest way to do a seed sweep.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "REPORT_VERSION",
    "EvalPin",
    "EvalReport",
    "IncomparableReports",
    "capture_environment",
    "compare_reports",
    "format_comparison",
    "prompt_set_digest",
]

#: Bumped when the report layout changes in a way that breaks a reader.
REPORT_VERSION = 1


class IncomparableReports(ValueError):
    """Two evaluation reports were compared whose pinned settings differ.

    Carries the differing fields so the message says exactly which setting
    invalidates the comparison, and what to do about it.
    """

    def __init__(self, differences: Mapping[str, tuple[Any, Any]]) -> None:
        self.differences = dict(differences)
        detail = "; ".join(
            f"{key}: {first!r} vs {second!r}"
            for key, (first, second) in sorted(self.differences.items())
        )
        super().__init__(
            "refusing to compare evaluation reports with different pinned "
            f"settings — {detail}. Re-run one of them under the other's "
            "settings, or pass allow=(...) naming the field you are "
            "deliberately varying. A table that mixes these is not a "
            "comparison of the models."
        )


def prompt_set_digest(prompts: Sequence[str]) -> str:
    """Return a stable content hash identifying a prompt set.

    Order-sensitive on purpose. Two runs over the same prompts in a different
    order draw different noise per prompt when the seed is fixed, so they are
    not the same evaluation.

    Args:
        prompts: The prompts, in evaluation order.

    Returns:
        A 16-character hex digest, plus the count, e.g. ``"50:a3f1c0..."``.
    """
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\x00")
    return f"{len(prompts)}:{digest.hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class EvalPin:
    """Everything that must match before two reports may be compared.

    Args:
        checkpoint: Identifier of the weights evaluated. Not itself required to
            match — comparing two checkpoints is the whole point — but recorded
            so a report identifies its subject.
        sampler: Sampler name.
        steps: Denoising steps.
        guidance: Classifier-free guidance scale.
        guidance_rescale: CFG-rescale factor.
        negative_prompt: The unconditional branch's prompt. An empty string and
            a hand-tuned negative prompt are different experiments.
        seed: Sampling seed.
        num_samples: Samples the statistics were computed over.
        prompt_digest: Content hash of the prompt set, from
            :func:`prompt_set_digest`.
        frames: Frames generated per sample.
        height: Latent or pixel rows generated.
        width: Latent or pixel columns generated.
        decoded: Whether metrics ran on decoded pixels or on latents. The same
            metric name means a different number in each space.
        codec: Fingerprint of the decoder, when decoded. Two VAEs give
            different pixel statistics from the same latents.
        precision: Inference dtype.
        metrics: Metric name to its configured parameters, rendered as a
            string. A changed threshold is a changed metric.

    Raises:
        ValueError: If a count is non-positive.
    """

    checkpoint: str = ""
    sampler: str = "euler"
    steps: int = 30
    guidance: float = 5.0
    guidance_rescale: float = 0.0
    negative_prompt: str = ""
    seed: int = 0
    num_samples: int = 0
    prompt_digest: str = ""
    frames: int = 0
    height: int = 0
    width: int = 0
    decoded: bool = False
    codec: str = ""
    precision: str = "bfloat16"
    metrics: dict[str, str] = field(default_factory=dict)

    #: Fields allowed to differ without invalidating a comparison. Declared
    #: ``ClassVar`` so it is a constant and not a pinned field — an annotated
    #: attribute on a dataclass silently becomes a constructor argument, and a
    #: pin whose exemption list is per-instance exempts nothing reliably.
    #: ``checkpoint`` is here because comparing checkpoints is the point.
    #: ``num_samples`` is deliberately NOT — see the module docstring.
    COMPARABLE_EXEMPT: ClassVar[tuple[str, ...]] = ("checkpoint",)

    def __post_init__(self) -> None:
        """Validate the counts recorded in the pin."""
        if self.steps < 1:
            raise ValueError(f"EvalPin.steps must be >= 1; got {self.steps!r}")
        if self.num_samples < 0:
            raise ValueError(
                f"EvalPin.num_samples must be >= 0; got {self.num_samples!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the pin.

        Returns:
            Field name to value.
        """
        payload = asdict(self)
        payload["metrics"] = dict(sorted(self.metrics.items()))
        return payload

    def differences(
        self,
        other: EvalPin,
        *,
        allow: Iterable[str] = (),
    ) -> dict[str, tuple[Any, Any]]:
        """Return the pinned fields that differ, excluding exempt ones.

        Args:
            other: The pin to compare against.
            allow: Additional field names permitted to differ, for a
                deliberate one-variable sweep.

        Returns:
            Field name to ``(this, other)``.
        """
        exempt = set(self.COMPARABLE_EXEMPT) | set(allow)
        first, second = self.to_dict(), other.to_dict()
        return {
            key: (first[key], second[key])
            for key in first
            if key not in exempt and first[key] != second[key]
        }

    def describe(self) -> str:
        """Return a one-line human summary of the sampling settings.

        Returns:
            A compact description suitable for a table caption — which is
            exactly where it should go, so the caption states what the numbers
            are.
        """
        space = "pixels" if self.decoded else "latents"
        return (
            f"{self.sampler} x{self.steps} cfg={self.guidance} seed={self.seed} "
            f"{self.frames}x{self.height}x{self.width} {space} "
            f"n={self.num_samples} prompts={self.prompt_digest or 'unset'}"
        )


@dataclass(frozen=True, slots=True)
class EvalReport:
    """A JSON-safe evaluation result carrying the settings that produced it.

    Args:
        pin: The settings all comparisons are gated on.
        metrics: Metric key to value. Keys are ``<metric>/<statistic>``, as
            produced by :meth:`avgen.eval.protocols.RunningMetric.compute`.
        skipped: Metric name to the reason it produced no number — a missing
            backend, an input the run did not have, or a pixel-only metric
            evaluated in latent space. Recorded rather than dropped, because a
            silently absent row reads as "not measured" when it usually means
            "could not be measured".
        run_name: The training run being evaluated.
        step: Training step of the evaluated checkpoint.
        external: Metric values computed outside avgen, kept separate so a
            reader can see which numbers this code produced and which it did
            not.
        environment: Versions and hardware, captured automatically.
        created_at: ISO-8601 UTC timestamp.
        report_version: Layout version.
        notes: Free text.
    """

    pin: EvalPin
    metrics: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    run_name: str = ""
    step: int = 0
    external: dict[str, float] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    report_version: int = REPORT_VERSION
    notes: str = ""

    def __post_init__(self) -> None:
        """Stamp the timestamp and environment when they were not supplied."""
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(UTC).isoformat(timespec="seconds")
            )
        if not self.environment:
            object.__setattr__(self, "environment", capture_environment())

    def to_dict(self) -> dict[str, Any]:
        """Return the whole report as JSON-safe containers.

        Returns:
            A mapping ready for :func:`json.dumps`.
        """
        return {
            "report_version": self.report_version,
            "run_name": self.run_name,
            "step": self.step,
            "created_at": self.created_at,
            "pin": self.pin.to_dict(),
            "pin_summary": self.pin.describe(),
            "metrics": dict(sorted(self.metrics.items())),
            "external": dict(sorted(self.external.items())),
            "skipped": dict(sorted(self.skipped.items())),
            "environment": dict(sorted(self.environment.items())),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvalReport:
        """Rebuild a report from :meth:`to_dict` output.

        Args:
            payload: A previously serialised report.

        Returns:
            The report.

        Raises:
            ValueError: If the payload comes from a newer report version, which
                may carry pinned fields this code would ignore — and ignoring a
                pinned field is precisely the failure the pin exists to prevent.
        """
        version = int(payload.get("report_version", 0))
        if version > REPORT_VERSION:
            raise ValueError(
                f"report_version={version} is newer than this avgen understands "
                f"({REPORT_VERSION}); an unknown pinned field would be silently "
                "ignored, so the comparison it guards cannot be trusted"
            )
        pin_payload = dict(payload.get("pin", {}))
        pin_payload.pop("COMPARABLE_EXEMPT", None)
        return cls(
            pin=EvalPin(**pin_payload),
            metrics=dict(payload.get("metrics", {})),
            skipped=dict(payload.get("skipped", {})),
            run_name=str(payload.get("run_name", "")),
            step=int(payload.get("step", 0)),
            external=dict(payload.get("external", {})),
            environment=dict(payload.get("environment", {})),
            created_at=str(payload.get("created_at", "")),
            report_version=version or REPORT_VERSION,
            notes=str(payload.get("notes", "")),
        )

    def save(self, path: str | Path) -> Path:
        """Write the report as indented JSON.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The written path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        """Read a report written by :meth:`save`.

        Args:
            path: Report file.

        Returns:
            The report.

        Raises:
            ValueError: If the file is not a readable report.
        """
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read evaluation report {source}: {error}"
            ) from error
        return cls.from_dict(payload)

    def with_metrics(self, values: Mapping[str, float]) -> EvalReport:
        """Return a copy with additional metric values folded in.

        Args:
            values: Metric key to value.

        Returns:
            A new report.
        """
        merged = dict(self.metrics)
        merged.update(values)
        return replace(self, metrics=merged)

    def render(self) -> str:
        """Render the report for a terminal.

        Returns:
            A multi-line string with the pin above the numbers, in that order
            deliberately: the settings are what make the numbers mean anything.
        """
        lines = [
            "=" * 72,
            f"avgen evaluation — {self.run_name or '(unnamed run)'} step {self.step}",
            "=" * 72,
            f"  checkpoint  {self.pin.checkpoint or '(unspecified)'}",
            f"  settings    {self.pin.describe()}",
            "",
        ]
        if self.metrics:
            width = max(len(key) for key in self.metrics)
            lines.append("METRICS")
            for key, value in sorted(self.metrics.items()):
                lines.append(f"    {key.ljust(width)}  {value: .6f}")
        else:
            lines.append("METRICS  (none computed)")
        if self.external:
            lines += ["", "EXTERNAL (computed outside avgen)"]
            for key, value in sorted(self.external.items()):
                lines.append(f"    {key}  {value: .6f}")
        if self.skipped:
            lines += ["", "SKIPPED"]
            for key, reason in sorted(self.skipped.items()):
                lines.append(f"    {key}: {reason}")
        lines.append("=" * 72)
        return "\n".join(lines)


def capture_environment() -> dict[str, str]:
    """Capture versions and hardware for the report's provenance section.

    Deliberately tolerant: this runs on a machine that may have no GPU and a
    partially installed avgen, and a failure to record the environment must not
    fail an evaluation.

    Returns:
        Key to value, with absent items simply omitted.
    """
    environment: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        environment["torch"] = torch.__version__
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            environment["cuda"] = torch.version.cuda or "unknown"
            environment["device"] = torch.cuda.get_device_name(0)
    except Exception:
        environment["torch"] = "unavailable"
    return environment


def compare_reports(
    a: EvalReport,
    b: EvalReport,
    *,
    allow: Iterable[str] = (),
) -> dict[str, dict[str, float]]:
    """Compare two evaluation reports, refusing when their pins differ.

    Args:
        a: The baseline report.
        b: The candidate report.
        allow: Pinned field names permitted to differ. Use it for a deliberate
            one-variable study — ``allow=("seed",)`` for a seed sweep,
            ``allow=("steps",)`` for a step-count sweep — and for nothing else.

    Returns:
        Metric key to ``{"a", "b", "delta", "relative"}``. Keys present in only
        one report appear with the other side omitted from the deltas.

    Raises:
        IncomparableReports: If any non-exempt pinned field differs. This is
            the whole point of the module: most published video-generation
            comparisons are invalid for exactly this reason, and an API that
            makes the invalid comparison inconvenient is the only reliable fix.
    """
    differences = a.pin.differences(b.pin, allow=allow)
    if differences:
        raise IncomparableReports(differences)

    keys = sorted(set(a.metrics) | set(b.metrics))
    comparison: dict[str, dict[str, float]] = {}
    for key in keys:
        row: dict[str, float] = {}
        if key in a.metrics:
            row["a"] = a.metrics[key]
        if key in b.metrics:
            row["b"] = b.metrics[key]
        if "a" in row and "b" in row:
            row["delta"] = row["b"] - row["a"]
            row["relative"] = row["delta"] / row["a"] if row["a"] else float("inf")
        comparison[key] = row
    return comparison


def format_comparison(
    comparison: Mapping[str, Mapping[str, float]],
    *,
    left: str = "baseline",
    right: str = "candidate",
) -> str:
    """Render a comparison as an aligned table.

    Args:
        comparison: Output of :func:`compare_reports`.
        left: Label for the first report.
        right: Label for the second.

    Returns:
        A multi-line string.
    """
    if not comparison:
        return "no metrics in common"
    width = max(len(key) for key in comparison)
    header = (
        f"{'metric'.ljust(width)}  {left:>14}  {right:>14}  "
        f"{'delta':>14}  {'relative':>10}"
    )
    lines = [header, "-" * len(header)]
    for key, row in comparison.items():
        first = f"{row['a']:14.6f}" if "a" in row else " " * 14
        second = f"{row['b']:14.6f}" if "b" in row else " " * 14
        delta = f"{row['delta']:14.6f}" if "delta" in row else " " * 14
        relative = f"{row['relative']:9.2%}" if "relative" in row else " " * 10
        lines.append(f"{key.ljust(width)}  {first}  {second}  {delta}  {relative}")
    return "\n".join(lines)
