"""Model architectures and the registry that builds them from a name.

Importing this package is what registers the built-in architectures, so a
config file naming ``video_dit`` or ``av_dit`` resolves without anything else
being imported first. Third-party architectures register themselves through
``importlib.metadata`` entry points in the ``avgen.models`` group; see
:mod:`avgen.models.registry`.

The layering inside this package is strict and worth preserving:
``attention`` and ``rope`` know nothing about models, ``layers`` composes them
into primitives, ``blocks`` composes those into the repeated unit the
parallelism layer addresses by name, and ``dit`` assembles blocks into a model.
Nothing imports upwards.
"""

from avgen.models.attention import (
    attention,
    attention_backends,
    device_arch,
    key_padding_mask,
    sdpa_context,
)
from avgen.models.blocks import (
    CrossModalFusion,
    DiTBlock,
    TextRefinerBlock,
    temporal_neighbour_mask,
)
from avgen.models.dit import (
    FUSION_MODES,
    AVDiT,
    AVDiTConfig,
    VideoDiT,
    VideoDiTConfig,
    list_presets,
    preset,
)
from avgen.models.layers import (
    MODULATION_CHUNKS,
    AdaLNModulation,
    CrossAttention,
    SelfAttention,
    SwiGLU,
    TimestepEmbedding,
    init_linear,
    init_norm,
    modulate,
    rms_norm,
    sinusoidal_embedding,
)
from avgen.models.registry import (
    MODEL_ENTRY_POINT_GROUP,
    build_model,
    list_models,
    model_config_class,
    register_model,
)
from avgen.models.rope import (
    ROPE_SCALING_MODES,
    RoPEScaling,
    RotaryEmbedding,
    RotaryTables,
    apply_rotary,
    build_rotary_tables,
    rope_axis_pairs,
)

__all__ = [
    "FUSION_MODES",
    "MODEL_ENTRY_POINT_GROUP",
    "MODULATION_CHUNKS",
    "ROPE_SCALING_MODES",
    "AVDiT",
    "AVDiTConfig",
    "AdaLNModulation",
    "CrossAttention",
    "CrossModalFusion",
    "DiTBlock",
    "RoPEScaling",
    "RotaryEmbedding",
    "RotaryTables",
    "SelfAttention",
    "SwiGLU",
    "TextRefinerBlock",
    "TimestepEmbedding",
    "VideoDiT",
    "VideoDiTConfig",
    "apply_rotary",
    "attention",
    "attention_backends",
    "build_model",
    "build_rotary_tables",
    "device_arch",
    "init_linear",
    "init_norm",
    "key_padding_mask",
    "list_models",
    "list_presets",
    "model_config_class",
    "modulate",
    "preset",
    "register_model",
    "rms_norm",
    "rope_axis_pairs",
    "sdpa_context",
    "sinusoidal_embedding",
    "temporal_neighbour_mask",
]
