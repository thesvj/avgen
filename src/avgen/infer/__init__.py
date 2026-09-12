"""Inference: schedules, samplers, guidance, conditioning, and the pipeline.

The subsystem is layered so that each piece is usable without the ones above it:

* :mod:`avgen.infer.schedule` — the noise levels to step through. Pure tensors.
  **Read the module warning about matching the training shift.**
* :mod:`avgen.infer.sampler` — one ODE step as a function of the model output.
  Knows nothing about models or prompts, so a training-time validation sample and
  a production sample run the identical solver.
* :mod:`avgen.infer.guidance` — classifier-free guidance, CFG-rescale, adaptive
  projected guidance, modality-aware composition, and scale schedules.
* :mod:`avgen.infer.conditioning` — builds a
  :class:`~avgen.core.model_input.ModelInput` for each inference task through the
  *same* three-step recipe the training objective uses. This is where the absence
  of train/inference skew actually lives.
* :mod:`avgen.infer.pipeline` — sequencing only: encode, loop, decode, and record
  the exact config that produced the sample.

Nothing here imports :mod:`avgen.models`, :mod:`avgen.train`, or
:mod:`avgen.checkpoint` at module scope; the pipeline takes any callable
satisfying the ``ModelInput -> ModelOutput`` ABI.
"""

from avgen.infer.conditioning import (
    StreamState,
    build_model_input,
    condition_first_frame,
    condition_mask,
    condition_stream,
    condition_temporal_prefix,
    default_audio_patchifier,
    first_frame_mask,
    temporal_prefix_mask,
    unconditional_input,
)
from avgen.infer.guidance import (
    GuidanceConfig,
    adaptive_projected_guidance,
    apply_guidance,
    apply_modality_guidance,
    classifier_free_guidance,
    rescale_guidance,
    unconditional_branch,
)
from avgen.infer.pipeline import (
    GeneratedMedia,
    GenerationConfig,
    GenerationPipeline,
)
from avgen.infer.sampler import (
    DPMSolverPlusPlus2M,
    EulerAncestralSampler,
    EulerSampler,
    HeunSampler,
    ResMultistepSampler,
    Sampler,
    SamplerConfig,
    build_sampler,
    list_samplers,
    register_sampler,
    to_epsilon,
    to_x0,
)
from avgen.infer.schedule import (
    ScheduleConfig,
    SigmaSchedule,
    apply_shift,
    build_sigma_schedule,
    karras_sigmas,
    linear_quadratic_sigmas,
    linear_sigmas,
    list_sigma_schedules,
    register_sigma_schedule,
    resolution_shift,
)

__all__ = [
    "DPMSolverPlusPlus2M",
    "EulerAncestralSampler",
    "EulerSampler",
    "GeneratedMedia",
    "GenerationConfig",
    "GenerationPipeline",
    "GuidanceConfig",
    "HeunSampler",
    "ResMultistepSampler",
    "Sampler",
    "SamplerConfig",
    "ScheduleConfig",
    "SigmaSchedule",
    "StreamState",
    "adaptive_projected_guidance",
    "apply_guidance",
    "apply_modality_guidance",
    "apply_shift",
    "build_model_input",
    "build_sampler",
    "build_sigma_schedule",
    "classifier_free_guidance",
    "condition_first_frame",
    "condition_mask",
    "condition_stream",
    "condition_temporal_prefix",
    "default_audio_patchifier",
    "first_frame_mask",
    "karras_sigmas",
    "linear_quadratic_sigmas",
    "linear_sigmas",
    "list_samplers",
    "list_sigma_schedules",
    "register_sampler",
    "register_sigma_schedule",
    "rescale_guidance",
    "resolution_shift",
    "temporal_prefix_mask",
    "to_epsilon",
    "to_x0",
    "unconditional_branch",
    "unconditional_input",
]
