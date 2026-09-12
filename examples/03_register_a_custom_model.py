"""Plug your own architecture in without touching the framework.

The whole model ABI is one method:

    forward(ModelInput) -> ModelOutput

Satisfy that, take a config dataclass as your first constructor argument (that
is what makes a YAML mapping buildable), expose `.blocks` as an
`nn.ModuleList`, and declare a tensor parallel plan — and your architecture
gets FSDP2, context parallelism, tensor parallelism, activation checkpointing,
sharded checkpoints, EMA, LoRA and the whole training and RL stack for free,
because none of those know or care what is inside a block.

Run:
    python examples/03_register_a_custom_model.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from avgen.core.model_input import ModelInput, ModelOutput
from avgen.core.patchify import GridPatchifier
from avgen.core.tokens import TextContext, TokenStream
from avgen.models.layers import FusedRMSNorm
from avgen.models.registry import build_model, list_models, register_model
from avgen.parallel.tensor import standard_block_plan, standard_root_plan


class MinimalBlock(nn.Module):
    """One transformer block, named the way the parallel plans expect.

    The submodule names are load-bearing: `standard_block_plan()` addresses
    `attention.q_proj`, `feed_forward.gate_proj` and so on by name, so a block
    that calls them something else silently gets no tensor parallelism.
    """

    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_norm = FusedRMSNorm(width)
        self.attention = nn.Module()
        self.attention.q_proj = nn.Linear(width, width, bias=False)
        self.attention.k_proj = nn.Linear(width, width, bias=False)
        self.attention.v_proj = nn.Linear(width, width, bias=False)
        self.attention.out_proj = nn.Linear(width, width, bias=False)
        self.ffn_norm = FusedRMSNorm(width)
        self.feed_forward = nn.Module()
        self.feed_forward.gate_proj = nn.Linear(width, 4 * width, bias=False)
        self.feed_forward.up_proj = nn.Linear(width, 4 * width, bias=False)
        self.feed_forward.down_proj = nn.Linear(4 * width, width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Attention then feed-forward, both pre-norm and residual."""
        batch, length, width = tokens.shape
        head_dim = width // self.num_heads

        normed = self.attention_norm(tokens)
        query, key, value = (
            projection(normed)
            .view(batch, length, self.num_heads, head_dim)
            .transpose(1, 2)
            for projection in (
                self.attention.q_proj,
                self.attention.k_proj,
                self.attention.v_proj,
            )
        )
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.attention.out_proj(attended)

        normed = self.ffn_norm(tokens)
        gated = torch.nn.functional.silu(self.feed_forward.gate_proj(normed))
        return tokens + self.feed_forward.down_proj(
            gated * self.feed_forward.up_proj(normed)
        )


@dataclass(frozen=True, slots=True)
class MinimalDiTConfig:
    """Everything the architecture needs, as one dataclass.

    The registry builds this from the `model:` block of a YAML config and
    ignores fields the architecture does not declare, so a model is free to
    interpret the shared schema its own way — or add fields of its own.
    """

    depth: int = 2
    width: int = 64
    num_heads: int = 4
    in_channels: int = 4
    patch_frames: int = 1
    patch_height: int = 2
    patch_width: int = 2


@register_model("minimal_dit")
class MinimalDiT(nn.Module):
    """A model is anything that maps a noisy TokenStream to a prediction."""

    config_class = MinimalDiTConfig

    def __init__(self, config: MinimalDiTConfig) -> None:
        super().__init__()
        depth, width, num_heads = config.depth, config.width, config.num_heads
        patch_dim = (
            config.in_channels
            * config.patch_frames
            * config.patch_height
            * config.patch_width
        )
        self.patch_embed = nn.Linear(patch_dim, width, bias=False)
        self.time_embed = nn.Linear(1, width, bias=False)
        # `.blocks` must be an nn.ModuleList: FSDP, activation checkpointing and
        # torch.compile all wrap per block, and they find the blocks by name.
        self.blocks = nn.ModuleList(
            MinimalBlock(width, num_heads) for _ in range(depth)
        )
        self.final_norm = FusedRMSNorm(width)
        self.final_proj = nn.Linear(width, patch_dim, bias=False)

    def tensor_parallel_plan(self, *, sequence_parallel: bool):
        """Reuse the standard plans; the submodule names above match them."""
        return (
            standard_root_plan(sequence_parallel=sequence_parallel),
            standard_block_plan(sequence_parallel=sequence_parallel),
        )

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Predict the per-token velocity field for the video stream."""
        video = inputs.video
        tokens = self.patch_embed(video.tokens)
        # Noise level is per-sample and per-modality: that independence is what
        # collapses text-to-video, image-to-video, continuation and inpainting
        # into one model rather than four.
        tokens = tokens + self.time_embed(video.noise_level[:, None, None].float())
        for block in self.blocks:
            tokens = block(tokens)
        # Predictions stay in token space: the objective compares them against
        # a token-space target, so unfolding to a grid here would be pure waste
        # in the hot loop. The sampler unfolds once, at the end.
        prediction = self.final_proj(self.final_norm(tokens))
        return ModelOutput(video=prediction, audio=inputs.audio.tokens)


def main() -> None:
    print("registered models:", list_models())

    # build_model takes the plain mapping a YAML file parses to.
    model = build_model("minimal_dit", {"depth": 2, "width": 64, "num_heads": 4})
    print(f"minimal_dit: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Build a token stream the way the patchifier does, so the coordinates and
    # layout are the real ones rather than a shape that happens to type-check.
    latents = torch.randn(2, 4, 4, 8, 8)
    patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
    layout = patchifier.layout_for(tuple(latents.shape))
    stream = patchifier.to_tokens(
        latents,
        # Physical time in seconds, one timestamp per latent frame — never an
        # index. That is what lets a sharded or packed sequence still know
        # where each token came from.
        positions=torch.arange(layout.frames).float().div(8.0).expand(2, layout.frames),
        mask=torch.ones(
            2, layout.frames, layout.height, layout.width, dtype=torch.bool
        ),
        noise_level=torch.rand(2),
    )
    print(f"tokens: {tuple(stream.tokens.shape)} from a {layout.grid} grid")

    output = model(
        ModelInput(
            video=stream,
            # A zero-length audio stream and an empty text context: the ABI is
            # the same whether or not a model uses them.
            audio=TokenStream.empty_like(2, width=stream.width),
            text=TextContext.empty(2, width=16),
        )
    )
    print("prediction:", tuple(output.video.shape))

    root, block = model.tensor_parallel_plan(sequence_parallel=True)
    print(f"tensor parallel plan: {len(root)} root entries, {len(block)} per block")
    print("\nThat is the entire contract. Everything else composes with it.")
    print(
        "To ship an architecture as a separate package, advertise it under the\n"
        "'avgen.models' entry-point group — avgen discovers it with no import here."
    )


if __name__ == "__main__":
    main()
