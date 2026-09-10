"""FLOP accounting, roofline time, and measured per-operator estimates.

Model FLOPs Utilisation (MFU) is the only honest way to compare two training
runs on different hardware, at different scales, with different parallelism. It
is also routinely computed wrong for video models, in two specific ways:

**The attention term is not negligible.** The usual language-model shortcut
``6 * params * tokens`` deliberately drops attention because at 4k context it is
a few percent. At 100k tokens it is the *majority* of the FLOPs — attention
scales with ``seq^2`` while the projections scale with ``seq`` — so dropping it
understates the work by several times and reports an MFU that is impossibly low.
:func:`transformer_flops` counts it explicitly.

**Activation checkpointing is real work.** Recomputation costs an extra forward
pass over the checkpointed blocks. It belongs in the denominator of hardware
utilisation and *not* in the numerator of model FLOPs — MFU measures useful
work, and recomputation is overhead. Reporting the two separately is what makes
"we enabled full AC and MFU dropped 8%" an interpretable sentence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from avgen.parallel.activation import ActivationCheckpointConfig
from avgen.parallel.dims import ParallelDims
from avgen.simulate.memory import ModelShape

__all__ = [
    "Accelerator",
    "ComputeEstimate",
    "estimate_compute",
    "measure_runtime",
    "transformer_flops",
]


@dataclass(frozen=True, slots=True)
class Accelerator:
    """Peak arithmetic throughput of one device.

    Args:
        name: Device label.
        bf16_tflops: Dense bf16 matmul peak in TFLOP/s. Use the *dense* number,
            not the sparsity-doubled marketing figure.
        fp8_tflops: Dense fp8 peak, when supported.
        memory_gib: Device memory.
        memory_bandwidth_tbps: HBM bandwidth in TB/s, used for the
            memory-bound side of the roofline.
    """

    name: str
    bf16_tflops: float
    memory_gib: float
    fp8_tflops: float = 0.0
    memory_bandwidth_tbps: float = 3.35

    def peak_flops(self, *, fp8: bool = False) -> float:
        """Return peak FLOP/s for the requested precision.

        Args:
            fp8: Whether to use the fp8 peak.

        Returns:
            FLOP/s.
        """
        tflops = self.fp8_tflops if fp8 and self.fp8_tflops else self.bf16_tflops
        return tflops * 1e12


#: Published dense peaks. Verify against your own silicon before trusting an
#: MFU number derived from them.
A100_80GB = Accelerator(
    "A100 80GB", bf16_tflops=312.0, memory_gib=80.0, memory_bandwidth_tbps=2.0
)
H100_SXM = Accelerator(
    "H100 SXM",
    bf16_tflops=989.0,
    fp8_tflops=1979.0,
    memory_gib=80.0,
    memory_bandwidth_tbps=3.35,
)
H200_SXM = Accelerator(
    "H200 SXM",
    bf16_tflops=989.0,
    fp8_tflops=1979.0,
    memory_gib=141.0,
    memory_bandwidth_tbps=4.8,
)
B200 = Accelerator(
    "B200",
    bf16_tflops=2250.0,
    fp8_tflops=4500.0,
    memory_gib=192.0,
    memory_bandwidth_tbps=8.0,
)


def transformer_flops(
    shape: ModelShape,
    *,
    include_backward: bool = True,
) -> dict[str, float]:
    """Count FLOPs for one sample through a pre-norm transformer.

    Args:
        shape: Model and batch geometry.
        include_backward: Whether to include the backward pass, which costs
            about twice the forward.

    Returns:
        FLOPs split into projection, attention, feed-forward, and cross-
        attention terms, plus the total.
    """
    seq = float(shape.sequence_length)
    width = float(shape.width)
    depth = float(shape.depth)
    head_dim = width / max(1, shape.num_heads)

    # q, k, v, out: four (seq, width) x (width, width) matmuls per block.
    projections = depth * 4.0 * 2.0 * seq * width * width

    # Scores (seq, seq, head_dim) and the value-weighted sum, both across heads.
    attention = depth * 2.0 * 2.0 * seq * seq * width

    # SwiGLU: gate and up out to mlp_ratio * width, down back.
    feed_forward = depth * 3.0 * 2.0 * seq * width * (width * shape.mlp_ratio)

    cross_attention = 0.0
    if shape.text_tokens:
        text = float(shape.text_tokens)
        # q from the sequence, k/v from text, plus the two attention matmuls.
        cross_attention = depth * (
            2.0 * 2.0 * seq * width * width
            + 2.0 * 2.0 * text * width * width
            + 2.0 * 2.0 * seq * text * width
        )

    forward = projections + attention + feed_forward + cross_attention
    multiplier = 3.0 if include_backward else 1.0
    batch = float(shape.micro_batch_size)
    return {
        "projections": projections * multiplier * batch,
        "attention": attention * multiplier * batch,
        "feed_forward": feed_forward * multiplier * batch,
        "cross_attention": cross_attention * multiplier * batch,
        "total": forward * multiplier * batch,
        "attention_fraction": attention / forward if forward else 0.0,
        "head_dim": head_dim,
    }


@dataclass(frozen=True, slots=True)
class ComputeEstimate:
    """Predicted arithmetic cost of one optimizer step.

    Args:
        model_flops: Useful FLOPs, excluding recomputation.
        hardware_flops: All FLOPs executed, including recomputation.
        seconds: Predicted compute time per rank per step.
        accelerator: Device the estimate is for.
        achieved_fraction: Fraction of peak the estimate assumes.
    """

    model_flops: float
    hardware_flops: float
    seconds: float
    accelerator: str
    achieved_fraction: float

    @property
    def recompute_overhead(self) -> float:
        """Extra FLOPs spent on recomputation, as a fraction of useful work."""
        if self.model_flops <= 0:
            return 0.0
        return (self.hardware_flops - self.model_flops) / self.model_flops

    def mfu(self, step_seconds: float, peak_flops: float) -> float:
        """Model FLOPs Utilisation given a measured step time.

        Args:
            step_seconds: Measured wall-clock seconds per optimizer step.
            peak_flops: Device peak FLOP/s.

        Returns:
            MFU in ``[0, 1]``.
        """
        if step_seconds <= 0 or peak_flops <= 0:
            return 0.0
        return self.model_flops / (step_seconds * peak_flops)

    def hfu(self, step_seconds: float, peak_flops: float) -> float:
        """Hardware FLOPs Utilisation, which counts recomputation.

        Args:
            step_seconds: Measured wall-clock seconds per optimizer step.
            peak_flops: Device peak FLOP/s.

        Returns:
            HFU in ``[0, 1]``.
        """
        if step_seconds <= 0 or peak_flops <= 0:
            return 0.0
        return self.hardware_flops / (step_seconds * peak_flops)


def estimate_compute(
    shape: ModelShape,
    dims: ParallelDims,
    *,
    accelerator: Accelerator = H100_SXM,
    activation_checkpoint: ActivationCheckpointConfig | None = None,
    achieved_fraction: float = 0.45,
    use_fp8: bool = False,
) -> ComputeEstimate:
    """Predict per-rank compute time for one optimizer step.

    ``achieved_fraction`` defaults to 0.45, which is a realistic MFU for a
    well-tuned large transformer — not the 0.8 a naive roofline would suggest.
    The gap is real: kernel launch gaps, non-matmul operations, memory-bound
    norms and elementwise work, and imperfect overlap. Calibrate it from a real
    run and every downstream projection improves.

    Args:
        shape: Model and batch geometry.
        dims: Parallelism degrees, which divide the per-rank work.
        accelerator: Device profile.
        activation_checkpoint: Policy, used for the recomputation term.
        achieved_fraction: Fraction of peak actually achieved.
        use_fp8: Whether to price against the fp8 peak.

    Returns:
        The estimate.
    """
    policy = activation_checkpoint or ActivationCheckpointConfig(mode="none")
    flops = transformer_flops(shape)
    model_flops = flops["total"]

    if policy.mode == "full":
        recompute_fraction = 1.0 / 3.0  # one extra forward on top of fwd+bwd
    elif policy.mode == "selective_layer":
        recompute_fraction = (1.0 / policy.layer_interval) / 3.0
    elif policy.mode == "selective_op":
        # Only the cheap operations are replayed, so the overhead is a small
        # fraction of a forward pass rather than all of it.
        recompute_fraction = 0.08
    else:
        recompute_fraction = 0.0

    hardware_flops = model_flops * (1.0 + recompute_fraction)

    # Every model-parallel axis divides this rank's share of the arithmetic.
    per_rank = hardware_flops / max(1, dims.tensor * dims.context * dims.pipeline)
    peak = accelerator.peak_flops(fp8=use_fp8)
    seconds = per_rank / (peak * achieved_fraction) if peak else 0.0

    return ComputeEstimate(
        model_flops=model_flops / max(1, dims.tensor * dims.context * dims.pipeline),
        hardware_flops=per_rank,
        seconds=seconds,
        accelerator=accelerator.name,
        achieved_fraction=achieved_fraction,
    )


def measure_runtime(
    step: Callable[[], Any],
    *,
    mode: Literal["operator-level-benchmark", "operator-level-cost-model"] = (
        "operator-level-cost-model"
    ),
) -> dict[str, float]:
    """Estimate step time per operator using PyTorch's runtime estimator.

    Two modes, with a real trade-off:

    ``operator-level-cost-model``
        Analytical, runs under fake tensors, needs no GPU. Fast enough to sweep,
        accurate to roughly the right order.

    ``operator-level-benchmark``
        Actually times each operator on the device. Far more accurate, needs a
        real GPU, and takes as long as running the step several times.

    Args:
        step: Zero-argument callable running one forward and backward.
        mode: Which estimator to use.

    Returns:
        Total estimated milliseconds and the per-operator breakdown.
    """
    from torch.distributed._tools.runtime_estimator import RuntimeEstimator

    estimator = RuntimeEstimator()
    with estimator(mode):
        step()
    return {
        "total_ms": float(estimator.total_runtime),
        **{
            str(name): float(value)
            for name, value in getattr(estimator, "mod_runtimes", {}).items()
        },
    }


def suggest_activation_policy(
    shape: ModelShape,
    dims: ParallelDims,
    *,
    accelerator: Accelerator = H100_SXM,
    headroom: float = 0.10,
) -> ActivationCheckpointConfig:
    """Choose the cheapest checkpointing policy that fits in device memory.

    Walks the policies from cheapest to most aggressive and returns the first
    that fits. This is the decision people otherwise make by launching a job,
    waiting for an OOM, and guessing again.

    Args:
        shape: Model and batch geometry.
        dims: Parallelism degrees.
        accelerator: Device profile.
        headroom: Fraction of device memory to keep free.

    Returns:
        The cheapest policy that fits, or ``full`` if none does — in which case
        the configuration needs more parallelism, not more checkpointing.
    """
    from avgen.simulate.memory import estimate_memory

    candidates = [
        ActivationCheckpointConfig(mode="none"),
        ActivationCheckpointConfig(mode="selective_op", save_op_frequency=1),
        ActivationCheckpointConfig(mode="selective_op", save_op_frequency=2),
        ActivationCheckpointConfig(mode="selective_layer", layer_interval=2),
        ActivationCheckpointConfig(mode="full"),
    ]
    for policy in candidates:
        estimate = estimate_memory(shape, dims, activation_checkpoint=policy)
        if estimate.fits_in(accelerator.memory_gib, headroom=headroom):
            return policy
    return ActivationCheckpointConfig(mode="full")
