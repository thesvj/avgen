"""Gated learned metrics: declared, probed, and never silently substituted.

FVD, FID, CLIPScore, VideoScore and HPS all need a pretrained network. avgen
will not pull one in as a core dependency, so every one of them is *declared*
here with its required extra and its required weights, probed at call time, and
refused with a precise, actionable message when it cannot be built.

**The rule this module exists to enforce: never substitute.** If FVD is
requested and its backend is missing, the answer is an error, not a different
metric under the FVD label, and not a silently skipped row. A comparison table
where one column was quietly computed a different way is worse than a table with
a hole in it, because the hole is visible.

**Why FVD numbers are not comparable across papers, and what avgen does about
it.** The Fréchet Video Distance is a Fréchet distance between Gaussian fits to
feature distributions, and every one of the following changes the number by more
than the differences typically being reported:

* which feature network (I3D vs VideoMAE vs a specific I3D checkpoint),
* the number of samples (the estimator is biased and the bias falls with N),
* frame count, resolution, and the resize filter used to reach them,
* whether real and generated clips were preprocessed identically.

So :class:`FrechetVideoDistance` requires the feature extractor to be passed in
explicitly — there is no default — and records its fingerprint in the report.
A number produced here is comparable with another number produced here under
the same pin, and with nothing else. That is the most any implementation can
honestly offer.

The Fréchet distance itself is implemented in this module with no dependency at
all (:func:`frechet_distance`), because the mathematics is fifteen lines and
importing scipy for a matrix square root is not worth a dependency.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from avgen.eval.protocols import MetricError

__all__ = [
    "LearnedMetricSpec",
    "MissingBackendError",
    "available_learned_metrics",
    "build_learned_metric",
    "describe_learned_metrics",
    "frechet_distance",
    "learned_metric_available",
    "learned_metric_summary",
    "list_learned_metrics",
    "merge_feature_banks",
    "register_learned_metric",
    "require_backend",
]


class MissingBackendError(RuntimeError):
    """A learned metric was requested whose backend is not installed.

    A ``RuntimeError`` rather than an ``ImportError`` because the fix is an
    install command, not a code change, and the message carries that command.
    """


@dataclass(frozen=True, slots=True)
class LearnedMetricSpec:
    """Declaration of a metric that needs a pretrained network.

    Args:
        name: Registry key, usable anywhere a built-in metric name is.
        extra: The avgen optional extra that provides the backend, e.g.
            ``"text"``. Empty when the requirement is a third-party package
            with no avgen extra.
        modules: Importable module names the backend needs. Probed with
            ``importlib.util.find_spec``, which does not execute them — a
            probe that imports torchvision costs a second and can fail on a
            headless node for reasons unrelated to whether it is installed.
        weights: Human description of the weights or model id the metric also
            needs. Presence of the package is necessary and not sufficient.
        measures: One line on what the metric actually measures.
        caveat: The reason a number from this metric is not comparable with a
            number from someone else's implementation.
        factory: Callable building the metric once its backend is confirmed.
            ``None`` means avgen declares the metric but ships no
            implementation, which is stated plainly rather than papered over.
    """

    name: str
    extra: str
    modules: tuple[str, ...]
    weights: str
    measures: str
    caveat: str
    factory: Callable[..., Any] | None = field(default=None, repr=False)

    def install_hint(self) -> str:
        """Return the exact command that makes this metric available.

        Returns:
            A pip invocation, plus the weights requirement.
        """
        if self.extra:
            command = f"pip install 'avgen[{self.extra}]'"
        else:
            command = f"pip install {' '.join(self.modules)}"
        return f"{command}  (and: {self.weights})"

    def missing_modules(self) -> tuple[str, ...]:
        """Return the declared modules that are not importable.

        Returns:
            Module names, in declaration order.
        """
        return tuple(
            module
            for module in self.modules
            if importlib.util.find_spec(module.split(".")[0]) is None
        )


_LEARNED: dict[str, LearnedMetricSpec] = {}


def register_learned_metric(spec: LearnedMetricSpec) -> LearnedMetricSpec:
    """Add a gated metric declaration to the registry.

    Args:
        spec: The declaration.

    Returns:
        The same spec, so this can be used at module scope.

    Raises:
        ValueError: If the name is already registered.
    """
    if spec.name in _LEARNED:
        raise ValueError(f"learned metric {spec.name!r} is already registered")
    _LEARNED[spec.name] = spec
    return spec


def list_learned_metrics() -> tuple[str, ...]:
    """Return every declared gated metric name, sorted.

    Returns:
        The names.
    """
    return tuple(sorted(_LEARNED))


def learned_metric_available(name: str) -> bool:
    """Whether a gated metric's declared backend modules are importable.

    Availability of the *package* does not guarantee the *weights* are present;
    that is checked when the metric is actually built.

    Args:
        name: Registry key.

    Returns:
        Whether every declared module can be found.

    Raises:
        MetricError: If the name is not a declared learned metric.
    """
    return not _spec(name).missing_modules()


def available_learned_metrics() -> dict[str, bool]:
    """Return every declared gated metric and whether its backend is present.

    Returns:
        Name to availability, sorted by name.
    """
    return {name: learned_metric_available(name) for name in list_learned_metrics()}


def describe_learned_metrics() -> dict[str, dict[str, str]]:
    """Return a human-readable table of the gated metrics.

    Returns:
        Name to a mapping with ``measures``, ``caveat``, ``requires``, and
        ``status``.
    """
    described: dict[str, dict[str, str]] = {}
    for name in list_learned_metrics():
        spec = _spec(name)
        missing = spec.missing_modules()
        described[name] = {
            "measures": spec.measures,
            "caveat": spec.caveat,
            "requires": spec.install_hint(),
            "status": "available" if not missing else f"missing {', '.join(missing)}",
        }
    return described


def _spec(name: str) -> LearnedMetricSpec:
    """Look up a declaration, or raise with the list of declared names."""
    if name not in _LEARNED:
        raise MetricError(
            f"{name!r} is not a declared learned metric; declared: "
            f"{', '.join(list_learned_metrics())}"
        )
    return _LEARNED[name]


def require_backend(name: str) -> LearnedMetricSpec:
    """Assert a gated metric's backend is importable, or refuse precisely.

    Args:
        name: Registry key.

    Returns:
        The declaration.

    Raises:
        MissingBackendError: If any declared module is absent. The message
            names the metric, the missing modules, and the install command.
        MetricError: If the name is not declared.
    """
    spec = _spec(name)
    missing = spec.missing_modules()
    if missing:
        raise MissingBackendError(
            f"metric {name!r} needs {', '.join(missing)}, which is not "
            f"installed. Install it with:  {spec.install_hint()}\n"
            "avgen will not substitute a different metric under this name: a "
            "table with one column computed differently is worse than a table "
            "with a hole in it."
        )
    return spec


def build_learned_metric(name: str, **kwargs: Any) -> Any:
    """Construct a gated metric, refusing clearly when it cannot be built.

    Args:
        name: Registry key.
        **kwargs: Constructor arguments, including any required weights or
            feature extractor.

    Returns:
        The metric instance.

    Raises:
        MissingBackendError: If the backend is not installed, or if avgen
            declares the metric but ships no implementation for it.
        MetricError: If the name is not declared.
    """
    spec = require_backend(name)
    if spec.factory is None:
        raise MissingBackendError(
            f"metric {name!r} is declared but avgen ships no implementation of "
            f"it. It measures: {spec.measures}. Supply your own via "
            "avgen.eval.learned.register_learned_metric(), or compute it with "
            "the reference implementation and record the result in the "
            "EvalReport's `external` section so the pin still travels with the "
            "number."
        )
    return spec.factory(**kwargs)


# ---------------------------------------------------------------------------
# Fréchet distance: the one piece of this that needs no dependency
# ---------------------------------------------------------------------------


def _symmetric_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """Principal square root of a symmetric positive semi-definite matrix.

    Via eigendecomposition rather than an iterative Denman-Beavers scheme:
    covariance matrices are symmetric by construction, ``eigh`` is stable for
    them, and negative eigenvalues arising from floating-point error are
    clamped to zero rather than producing complex numbers that then get their
    imaginary parts silently discarded — which is what most FID implementations
    do, and is a real source of cross-implementation disagreement.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    root = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * root.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def frechet_distance(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Fréchet distance between Gaussian fits to two feature sets.

    This is the shared core of FID, FVD, and every other "Fréchet X distance":
    ``||mu_a - mu_b||^2 + tr(S_a + S_b - 2 (S_a S_b)^(1/2))``. Computed in
    float64 throughout — the trace terms of a 2048-dimensional covariance are
    large and nearly cancel, and float32 loses several significant digits.

    **The estimator is biased downward in the number of samples**, strongly so
    below a few thousand. Two Fréchet numbers computed from different sample
    counts are not comparable, which is why :class:`avgen.eval.EvalReport`
    records the count in its pin.

    Args:
        features_a: ``(n, d)`` features from the first set.
        features_b: ``(m, d)`` features from the second set.
        epsilon: Ridge added to each covariance diagonal, which keeps the
            matrix square root defined when the sample count is close to or
            below the feature dimension.

    Returns:
        The distance, as a Python float.

    Raises:
        MetricError: If either set has fewer than two samples or the feature
            dimensions disagree.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise MetricError(
            f"frechet_distance expects (n, d) matrices; got "
            f"{tuple(features_a.shape)} and {tuple(features_b.shape)}"
        )
    if features_a.shape[1] != features_b.shape[1]:
        raise MetricError(
            f"feature dimensions differ: {features_a.shape[1]} vs "
            f"{features_b.shape[1]}; the two sets must come from the same "
            "extractor"
        )
    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        raise MetricError(
            "frechet_distance needs at least two samples per set to estimate a "
            f"covariance; got {features_a.shape[0]} and {features_b.shape[0]}"
        )

    first = features_a.detach().to(torch.float64)
    second = features_b.detach().to(torch.float64)
    mean_a, mean_b = first.mean(dim=0), second.mean(dim=0)
    dimension = first.shape[1]
    ridge = epsilon * torch.eye(dimension, dtype=torch.float64, device=first.device)
    cov_a = first.T.cov() + ridge
    cov_b = second.T.cov() + ridge

    root_a = _symmetric_sqrt(cov_a)
    middle = _symmetric_sqrt(root_a @ cov_b @ root_a)
    difference = (mean_a - mean_b).pow(2).sum()
    return float(
        difference + torch.trace(cov_a) + torch.trace(cov_b) - 2.0 * torch.trace(middle)
    )


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _build_clip_score(
    *,
    model_id: str = "openai/clip-vit-large-patch14",
    device: str = "cpu",
) -> Any:
    """Build a CLIPScore metric from a transformers CLIP checkpoint."""
    from avgen.eval._clip import CLIPScore

    return CLIPScore(model_id=model_id, device=device)


register_learned_metric(
    LearnedMetricSpec(
        name="fvd",
        extra="",
        modules=("torch",),
        weights="an I3D or VideoMAE feature extractor you supply explicitly",
        measures=(
            "Fréchet distance between video-feature distributions of real and "
            "generated clips"
        ),
        caveat=(
            "not comparable across implementations: the feature network, the "
            "sample count, the frame count, the resolution, and the resize "
            "filter each move the number more than most reported differences"
        ),
        factory=None,
    )
)

register_learned_metric(
    LearnedMetricSpec(
        name="fid",
        extra="",
        modules=("torchvision",),
        weights="InceptionV3 pool3 weights (downloaded by torchvision on first use)",
        measures="Fréchet distance between per-frame image-feature distributions",
        caveat=(
            "an image metric applied to video frames: it is blind to every "
            "temporal property, so a model that emits good stills in a bad "
            "order scores well"
        ),
        factory=None,
    )
)

register_learned_metric(
    LearnedMetricSpec(
        name="clip_score",
        extra="text",
        modules=("transformers",),
        weights="a CLIP checkpoint, e.g. openai/clip-vit-large-patch14",
        measures="cosine similarity between prompt and per-frame image embeddings",
        caveat=(
            "measures prompt-image agreement in CLIP's own space, which is "
            "saturated and gameable; it cannot distinguish two videos that both "
            "contain the prompted content"
        ),
        factory=_build_clip_score,
    )
)

register_learned_metric(
    LearnedMetricSpec(
        name="video_score",
        extra="text",
        modules=("transformers",),
        weights="a VideoScore reward checkpoint",
        measures="a learned proxy for human quality judgements of generated video",
        caveat=(
            "a reward model trained on one generation distribution; optimising "
            "against it, or evaluating a model far from its training "
            "distribution, produces numbers that do not track human preference"
        ),
        factory=None,
    )
)

register_learned_metric(
    LearnedMetricSpec(
        name="hps",
        extra="text",
        modules=("transformers",),
        weights="a Human Preference Score checkpoint",
        measures="a learned human-preference score over generated media",
        caveat=(
            "trained on image preference data; applied to video frames it "
            "ignores motion entirely"
        ),
        factory=None,
    )
)


def learned_metric_summary() -> str:
    """Render the gated-metric table for ``avgen info``.

    Returns:
        A multi-line string, one metric per line, with its status.
    """
    rows = describe_learned_metrics()
    if not rows:
        return "  (none declared)"
    width = max(len(name) for name in rows)
    lines: list[str] = []
    for name, detail in rows.items():
        lines.append(f"  {name.ljust(width)}  {detail['status']}")
        lines.append(f"  {' ' * width}  requires: {detail['requires']}")
    return "\n".join(lines)


def merge_feature_banks(
    banks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Concatenate per-rank feature matrices in a deterministic order.

    Args:
        banks: Rank label to ``(n, d)`` features.

    Returns:
        One ``(sum_n, d)`` matrix, with ranks concatenated in sorted label
        order so that the result does not depend on gather ordering. Fréchet
        distance is invariant to sample order, but reproducibility of the
        *intermediate artifact* is what lets a disagreement be debugged.

    Raises:
        MetricError: If the banks disagree in feature dimension.
    """
    if not banks:
        raise MetricError("merge_feature_banks was given no feature banks")
    ordered = [banks[key] for key in sorted(banks)]
    dimensions = {tensor.shape[-1] for tensor in ordered}
    if len(dimensions) != 1:
        raise MetricError(
            f"feature banks disagree in dimension: {sorted(dimensions)}"
        )
    return torch.cat(ordered, dim=0)
