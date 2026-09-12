"""Adapt a pretrained model with LoRA, then merge it away for inference.

LoRA here is implemented natively on DTensor rather than borrowed from `peft`,
which is what lets an adapter sit on a model that is already sharded five ways
without gathering it first. DoRA is the same call with one flag.

The subtlety worth knowing: adapters must be attached *after* tensor
parallelism and *before* FSDP wraps anything, or the adapter parameters end up
outside the sharding plan. `parallelize(..., adapt=...)` is the seam that
guarantees that ordering — this example uses the single-device path, but the
hook is where a real run puts it.

Run:
    python examples/04_lora_finetune.py
"""

from __future__ import annotations

import torch

from avgen import load_config
from avgen.config.resolve import model_kwargs
from avgen.core.model_input import ModelInput
from avgen.core.patchify import GridPatchifier
from avgen.core.tokens import TextContext, TokenStream
from avgen.finetune import (
    LoRAConfig,
    apply_lora,
    lora_parameters,
    mark_only_lora_trainable,
    merge_lora,
    trainable_summary,
)
from avgen.models.registry import build_model


def main() -> None:
    config = load_config(
        None,
        overrides=[
            "model.name=video_dit",
            "model.depth=2",
            "model.width=64",
            "model.num_heads=4",
        ],
    )
    torch.manual_seed(0)
    model = build_model(config.model.name, model_kwargs(config))
    base_parameters = sum(p.numel() for p in model.parameters())

    # rank 8 on the attention and MLP projections. DoRA (use_dora=True)
    # decomposes each update into magnitude and direction; it is a strictly
    # better fit at the same rank and costs one extra norm per forward.
    model = apply_lora(model, LoRAConfig(rank=8, alpha=16.0, use_dora=True))
    mark_only_lora_trainable(model)

    summary = trainable_summary(model)
    print(f"base parameters:      {base_parameters:,}")
    print(f"trainable parameters: {summary.trainable:,} ({summary.percentage:.2f}%)")
    print(f"adapter tensors:      {len(list(lora_parameters(model)))}")

    # One optimisation step, so the example proves the gradient actually
    # reaches the adapters and nothing else.
    latents = torch.randn(2, config.model.in_channels, 4, 8, 8)
    patchifier = GridPatchifier(
        patch_frames=config.model.patch_frames,
        patch_height=config.model.patch_height,
        patch_width=config.model.patch_width,
    )
    layout = patchifier.layout_for(tuple(latents.shape))
    stream = patchifier.to_tokens(
        latents,
        positions=torch.arange(layout.frames).float().div(8.0).expand(2, layout.frames),
        mask=torch.ones(
            2, layout.frames, layout.height, layout.width, dtype=torch.bool
        ),
        noise_level=torch.rand(2),
    )

    # A text-to-video model still passes an audio stream: a zero-length one
    # makes every audio path degenerate to a no-op, with not one
    # `if audio is None` anywhere in the model.
    inputs = ModelInput(
        video=stream,
        audio=TokenStream.empty_like(2, width=stream.width),
        text=TextContext.empty(2, width=config.data.text_width),
    )

    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-3)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model(inputs).video.square().mean().backward()
    optimizer.step()

    moved = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert all("lora" in name or "magnitude" in name for name in moved), moved
    print(f"parameters changed by the step: {len(moved)}, all of them adapter tensors")

    # Merging folds W + (alpha/rank) * B @ A back into the base weight, so the
    # exported model has no adapter at inference time and costs nothing extra.
    with torch.no_grad():
        adapted = model(inputs).video.clone()
    merged = merge_lora(model, strip=True)
    with torch.no_grad():
        merged_out = merged(inputs).video

    error = (adapted - merged_out).abs().max().item()
    print(f"merged output matches the adapted one to {error:.2e}")
    print(f"modules left after strip: {sum(1 for _ in merged.modules())}")


if __name__ == "__main__":
    main()
