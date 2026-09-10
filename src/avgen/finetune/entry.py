"""The config-driven entry point behind ``avgen finetune``.

The rest of :mod:`avgen.finetune` is primitives — ``apply_lora``,
``freeze_except``, ``ControlAdapter``. This module is the one place that turns a
:class:`~avgen.config.RunConfig` into a running fine-tune, and it exists so the
CLI does not have to know the ordering rules that make adapter training correct.

There are two of those rules and both are easy to get wrong:

**Adapters go in between tensor parallelism and FSDP.** By that point the base
weights are already :class:`~torch.distributed.tensor.DTensor` shards, so an
adapter can derive its own placements from them and contract on the same mesh
dimension. FSDP has not yet run, so the new parameters are still picked up and
gradient-reduced. Inject after FSDP and the adapter is unmanaged: every rank
learns a different one, and nothing reports it. The ``adapt`` hook on
:func:`~avgen.parallel.apply.parallelize` is the seam for exactly this.

**Base weights load before the adapter is trained, not after.** A fine-tune that
silently starts from random initialisation still produces a falling loss curve;
it just produces a worse model than the base it was supposed to improve. So a
missing or unreadable base checkpoint is fatal here rather than a warning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

    from avgen.config import RunConfig

__all__ = ["build_adapter_hook", "finetune_from_config"]

_LOG = logging.getLogger("avgen.finetune")


def build_adapter_hook(config: RunConfig) -> Any:
    """Return the model transformation this fine-tune mode requires.

    Args:
        config: The validated run configuration.

    Returns:
        A callable suitable for :func:`~avgen.parallel.apply.parallelize`'s
        ``adapt`` argument, or ``None`` when the mode trains the base weights
        directly and needs no transformation.

    Raises:
        ValueError: If the configured mode is unknown.
    """
    settings = config.finetune
    mode = settings.mode

    if mode == "full":
        return None

    if mode == "lora":
        from avgen.finetune.lora import LoRAConfig, apply_lora, mark_only_lora_trainable

        lora = LoRAConfig(
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
            dropout=settings.lora_dropout,
            target_modules=tuple(settings.lora_targets),
            use_dora=settings.use_dora,
        )

        def _apply_lora(model: nn.Module) -> nn.Module:
            adapted = apply_lora(model, lora)
            mark_only_lora_trainable(adapted)
            return adapted

        return _apply_lora

    if mode == "control":
        from avgen.finetune.adapters import ControlAdapter, ControlAdapterConfig
        from avgen.finetune.freeze import freeze_all

        control = ControlAdapterConfig(
            control_channels=settings.control_channels,
            num_blocks=settings.control_blocks,
            scale=settings.control_scale,
        )

        def _apply_control(model: nn.Module) -> nn.Module:
            # The base tower is frozen and the side tower is zero-initialised, so
            # the adapted model is an exact identity at step zero. Anything else
            # would perturb a base model that is presumed good.
            freeze_all(model)
            return ControlAdapter(model, control)

        return _apply_control

    raise ValueError(
        f"unknown finetune.mode {mode!r}; expected one of: full, lora, control"
    )


def _load_base_weights(path: str, model: nn.Module) -> None:
    """Load the base checkpoint into a parallelised model.

    Args:
        path: Checkpoint directory.
        model: The parallelised model.

    Raises:
        RuntimeError: If the checkpoint cannot be read. This is deliberately
            fatal: a fine-tune that starts from random weights still shows a
            falling loss and is very hard to notice.
    """
    from avgen.checkpoint.export import import_safetensors

    try:
        import_safetensors(path, model, strict=False)
    except Exception as error:
        raise RuntimeError(
            f"could not load finetune.base_checkpoint from {path!r}: {error}. "
            "A fine-tune from random initialisation still produces a falling "
            "loss curve, so this is fatal rather than a warning."
        ) from error


def finetune_from_config(config: RunConfig) -> None:
    """Run a fine-tune described entirely by a configuration.

    Args:
        config: The validated run configuration. ``config.finetune`` selects the
            mode and ``config.finetune.base_checkpoint`` names the weights to
            start from.

    Raises:
        RuntimeError: If a required subsystem is unavailable or the base
            checkpoint cannot be read.
        ValueError: If the configured fine-tune mode is unknown.
    """
    from avgen.checkpoint import load as load_checkpoint
    from avgen.cli._wiring import build_trainer_config, build_training_stack
    from avgen.finetune.freeze import freeze_matching, trainable_summary
    from avgen.train.trainer import Trainer

    settings = config.finetune
    stack = build_training_stack(config, adapt=build_adapter_hook(config))

    _load_base_weights(settings.base_checkpoint, stack.parallel.model)

    if settings.freeze_patterns:
        freeze_matching(stack.parallel.model, tuple(settings.freeze_patterns))

    summary = trainable_summary(stack.parallel.model)
    _LOG.info(
        "finetune mode=%s trainable=%s of %s parameters (%.2f%%)",
        settings.mode,
        f"{summary.trainable:,}",
        f"{summary.total:,}",
        summary.percentage,
    )
    if summary.trainable == 0:
        raise RuntimeError(
            "no parameters are trainable after applying the fine-tune mode and "
            "freeze patterns. Check finetune.freeze_patterns — a run with an "
            "empty parameter set optimises nothing and never errors."
        )

    # The optimizer built by the stack covers every parameter; after freezing,
    # rebuild it over what actually trains so the moment buffers are not
    # allocated for weights that never move.
    from avgen.train.optimizer import build_optimizer

    stack.state.optimizer = build_optimizer(
        stack.parallel.model,
        name=config.train.optimizer,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    from avgen.train.schedule import build_schedule

    stack.state.schedule = build_schedule(
        config.train.schedule,
        stack.state.optimizer,
        total_steps=config.train.steps,
        warmup_steps=config.train.warmup_steps,
    )

    if config.checkpoint.resume:
        load_checkpoint(config.checkpoint.resume, stack.state, parallel=stack.parallel)

    trainer = Trainer(
        stack.state,
        stack.objective,
        stack.parallel,
        build_trainer_config(
            config,
            accumulation=stack.gradient_accumulation,
            data_world=stack.data_world,
        ),
        logger=stack.logger,
    )
    try:
        trainer.fit(stack.source, total_steps=config.train.steps)
    finally:
        close = getattr(stack.logger, "close", None)
        if callable(close):
            close()
