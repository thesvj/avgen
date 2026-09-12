"""Evaluation: cheap always-on diagnostics, gated learned metrics, pinned reports.

Three ideas, in order of how much they matter.

**A metric is only meaningful next to its counterpart.** Every dependency-free
metric here is trivially maximisable on its own — a frozen frame wins temporal
consistency, white noise wins motion magnitude — so each docstring says what the
metric fails to measure, and the defaults ship in complementary pairs.

**A gated metric is refused, never substituted.** FVD without its feature
network raises an error naming the missing extra. It never silently becomes a
different number under the same label.

**A report that does not carry its sampling settings is not evidence.** Every
:class:`EvalReport` pins the seed, steps, guidance, sampler, prompt-set hash and
checkpoint it came from, and :func:`compare_reports` refuses to compare two
reports whose pins differ. Most published video-generation comparisons are
invalid for exactly that reason; here the invalid comparison raises.

Typical use::

    from avgen.eval import EvalBatch, EvalPin, run_eval_suite

    report = run_eval_suite(
        batches,
        metrics=("temporal_consistency", "motion_magnitude", "static_frames"),
        pin=EvalPin(checkpoint="step-40000", steps=30, guidance=5.0, seed=0),
        prompts=prompts,
    )
    print(report.render())
"""

from avgen.eval.av import AudioBandwidth, AudioSilence, AVSyncProxy
from avgen.eval.conditional import (
    FirstFrameFidelity,
    InpaintBoundaryConsistency,
    SeamContinuity,
)
from avgen.eval.learned import (
    LearnedMetricSpec,
    MissingBackendError,
    available_learned_metrics,
    build_learned_metric,
    describe_learned_metrics,
    frechet_distance,
    learned_metric_available,
    learned_metric_summary,
    list_learned_metrics,
    register_learned_metric,
    require_backend,
)
from avgen.eval.protocols import (
    Metric,
    MetricError,
    RunningMetric,
    build_metric,
    describe_metrics,
    list_metrics,
    metric_class,
    register_metric,
)
from avgen.eval.report import (
    EvalPin,
    EvalReport,
    IncomparableReports,
    capture_environment,
    compare_reports,
    format_comparison,
    prompt_set_digest,
)
from avgen.eval.suite import (
    DEFAULT_PROMPTS,
    EvalBatch,
    MetricBundle,
    build_metrics,
    run_eval_suite,
    shard_prompts,
)
from avgen.eval.video import (
    FlickerIndex,
    MotionMagnitude,
    SaturationClipping,
    SharpnessProxy,
    StaticFrameRatio,
    TemporalConsistency,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "AVSyncProxy",
    "AudioBandwidth",
    "AudioSilence",
    "EvalBatch",
    "EvalPin",
    "EvalReport",
    "FirstFrameFidelity",
    "FlickerIndex",
    "IncomparableReports",
    "InpaintBoundaryConsistency",
    "LearnedMetricSpec",
    "Metric",
    "MetricBundle",
    "MetricError",
    "MissingBackendError",
    "MotionMagnitude",
    "RunningMetric",
    "SaturationClipping",
    "SeamContinuity",
    "SharpnessProxy",
    "StaticFrameRatio",
    "TemporalConsistency",
    "available_learned_metrics",
    "build_learned_metric",
    "build_metric",
    "build_metrics",
    "capture_environment",
    "compare_reports",
    "describe_learned_metrics",
    "describe_metrics",
    "format_comparison",
    "frechet_distance",
    "learned_metric_available",
    "learned_metric_summary",
    "list_learned_metrics",
    "list_metrics",
    "metric_class",
    "prompt_set_digest",
    "register_learned_metric",
    "register_metric",
    "require_backend",
    "run_eval_suite",
    "shard_prompts",
]
