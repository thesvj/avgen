"""An exponential moving average that survives FSDP2, DTensor, and DCP.

Almost every published diffusion sample is drawn from EMA weights, not from the
raw training weights, and the gap is large enough to see without a metric. The
reason is structural rather than incidental: the flow-matching gradient is
estimated from one noise level per sample, so consecutive steps pull the weights
in directions that are individually noisy and only correct in expectation. The
average of the trajectory is a much better estimate of that expectation than any
point on it.

Three things make an EMA hard once the model is sharded, and this module exists
for all three:

* **The weights are** :class:`~torch.distributed.tensor.DTensor` **shards.** The
  average must be taken shard-wise, keeping the DTensor wrapper, so that the
  checkpoint layer can reshard it onto a different rank count later. Calling
  ``full_tensor()`` to average a materialised copy would work on a small model
  and out-of-memory on a real one.
* **The memory is real.** A second copy of a 10B-parameter model is 40 GB in
  fp32. Storing the average in bf16 halves that, and is usually what makes an
  EMA affordable at all — with an important caveat spelled out in
  :class:`ShardedEMA`.
* **It has to be checkpointable.** An EMA that is not saved makes a preempted
  run unreproducible in the only artefact anyone cares about: the weights the
  samples came from.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

__all__ = ["ShardedEMA"]

#: Key prefix for shadow tensors in the state dict. DCP flattens nested state
#: dicts by joining keys, so a flat mapping with an explicit prefix is what a
#: resharding load actually sees, and keeping it explicit means the key a
#: checkpoint stores is the key this module reads.
_SHADOW_PREFIX = "shadow."


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local view of a possibly-sharded tensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class ShardedEMA:
    """Exponential moving average of the model weights, sharding-aware.

    The update is the standard one, with two corrections that matter in a real
    run.

    **The decay ramps in.** With a fixed ``decay = 0.9999`` the average has a
    time constant of ten thousand steps, so for the first several thousand steps
    it is dominated by the random initialisation and is strictly worse than the
    live weights — evaluating from it early tells you nothing. The effective
    decay is therefore ``min(decay, (1 + step) / (10 + step))``, which starts
    near ``0.1`` and approaches the configured value from below. The average is
    then a useful estimate from roughly the first hundred steps onwards.

    **The arithmetic runs on local shards.** For each parameter the update
    touches ``.to_local()``, not the DTensor. Both would give the same numbers,
    but the DTensor path dispatches through the sharding propagation rules for
    every one of a few thousand tensors on every step, and it does not implement
    every in-place op for every placement. The stored shadow stays a DTensor, so
    :meth:`state_dict` still hands the checkpoint layer something it can
    reshard.

    **The bf16 storage trade-off, stated honestly.** bf16 has 8 significand
    bits, so its relative resolution is about ``2^-8 = 0.004``. An update moves
    the shadow by ``(1 - decay) * (param - shadow)``. Once training settles and
    ``|param - shadow|`` falls to a small fraction of ``|shadow|``, that
    increment can land below half an ulp of the stored value and round away
    entirely — the average silently stops moving. With ``decay = 0.999`` the
    increment stays above the rounding floor for a long time; with
    ``decay = 0.9999`` and a converged model it may not. bf16 is nonetheless the
    default because the failure is graceful (a slightly stale average, not a
    wrong one), because the sample-quality difference between a slightly stale
    EMA and an exact one is below the noise floor of a diffusion sample, and
    because the alternative is often no EMA at all. **Pass
    ``storage_dtype=torch.float32`` when the decay is above ``0.9995`` and the
    memory is available** — that is the configuration where the truncation
    actually bites.

    Args:
        model: The module to track. Only floating-point parameters are
            averaged; integer buffers and non-float parameters are left alone,
            because an average of a discrete quantity is not a valid value of
            it.
        decay: Target decay. The realised decay ramps up to this value.
        storage_dtype: Dtype of the stored average.
        warmup_steps: Steps during which the shadow tracks the weights exactly
            instead of averaging. Use it to skip a known-unstable warmup phase
            whose weights should not enter the average at all.
        update_every: Steps between updates. Raising it cuts the EMA's memory
            bandwidth cost proportionally; the decay is compounded so the
            effective time constant is unchanged.
    """

    __slots__ = (
        "_decay",
        "_num_updates",
        "_shadow",
        "_storage_dtype",
        "_update_every",
        "_warmup_steps",
    )

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        storage_dtype: torch.dtype = torch.bfloat16,
        warmup_steps: int = 0,
        update_every: int = 1,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay!r}")
        if isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative; got {warmup_steps!r}")
        if isinstance(update_every, bool) or update_every < 1:
            raise ValueError(f"update_every must be >= 1; got {update_every!r}")
        self._decay = decay
        self._storage_dtype = storage_dtype
        self._warmup_steps = warmup_steps
        self._update_every = update_every
        self._num_updates = 0
        self._shadow: dict[str, torch.Tensor] = {
            name: parameter.detach().clone().to(self._storage_dtype)
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point()
        }

    @property
    def num_updates(self) -> int:
        """Number of optimizer steps folded into the average so far."""
        return self._num_updates

    @property
    def decay(self) -> float:
        """The configured target decay."""
        return self._decay

    def effective_decay(self, step: int) -> float:
        """Return the decay actually applied at a step.

        Args:
            step: Number of updates already folded in.

        Returns:
            ``min(decay, (1 + step) / (10 + step))``, compounded by
            ``update_every`` so that skipping updates does not lengthen the
            effective time constant.
        """
        if step < self._warmup_steps:
            # Tracking, not averaging: nothing before warmup should influence
            # the average that later samples are drawn from.
            return 0.0
        ramped = min(self._decay, (1.0 + step) / (10.0 + step))
        return ramped**self._update_every

    def update(self, model: nn.Module) -> None:
        """Fold the current weights into the average.

        Call once per *optimizer* step, never per microbatch: the average is
        over the weight trajectory, and a gradient-accumulation microbatch does
        not move the weights.

        Args:
            model: The module being trained. Must have the same parameter names
                as the module the EMA was constructed from.

        Raises:
            KeyError: If the model has grown or lost a tracked parameter, which
                means the average no longer describes this model.
        """
        step = self._num_updates
        self._num_updates += 1
        if step % self._update_every != 0:
            return
        weight = 1.0 - self.effective_decay(step)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.is_floating_point():
                    continue
                shadow = self._shadow.get(name)
                if shadow is None:
                    raise KeyError(
                        f"parameter {name!r} is not tracked by this EMA; the model "
                        "changed shape after the average was created"
                    )
                source = _local(parameter.detach())
                target = _local(shadow)
                # lerp_ requires matching dtypes, and casting the *parameter*
                # down is the cheap direction: it is a read, whereas casting the
                # shadow up would allocate a full-size fp32 temporary per
                # parameter per step.
                target.lerp_(source.to(target.dtype), weight)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's weights with the average, in place.

        Destructive. Use :meth:`apply_to` unless the training weights are
        genuinely finished with.

        Args:
            model: The module to overwrite.

        Raises:
            KeyError: If a tracked parameter is missing from ``model``.
        """
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.is_floating_point():
                    continue
                shadow = self._shadow.get(name)
                if shadow is None:
                    raise KeyError(f"parameter {name!r} is not tracked by this EMA")
                _local(parameter).copy_(_local(shadow).to(parameter.dtype))

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily swap the averaged weights in for evaluation.

        The training weights are held in a backup at their own dtype for the
        duration, so the peak cost of an evaluation is one extra copy of the
        model. That is the honest price of evaluating mid-run without a second
        process; the alternative — evaluating from a separate rank group holding
        its own copy — costs the same memory permanently.

        Args:
            model: The module to evaluate.

        Yields:
            The same module, with EMA weights installed.
        """
        backup = {
            name: _local(parameter).detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point()
        }
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    saved = backup.get(name)
                    if saved is not None:
                        _local(parameter).copy_(saved)

    def state_dict(self) -> dict[str, Any]:
        """Return the average and its progress counter.

        The shadow tensors are returned as they are stored — DTensors stay
        DTensors — so ``torch.distributed.checkpoint`` can save them sharded and
        reload them onto a different rank count.

        Returns:
            A flat mapping of shadow tensors plus scalar metadata.
        """
        state: dict[str, Any] = {
            f"{_SHADOW_PREFIX}{name}": tensor for name, tensor in self._shadow.items()
        }
        state["num_updates"] = self._num_updates
        state["decay"] = self._decay
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the average in place.

        Tensors are copied into the existing shadow buffers rather than
        replacing them, so a DCP load that has already resharded into those
        buffers stays valid and the placement is never silently changed.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: If a tracked parameter is missing from ``state``.
        """
        with torch.no_grad():
            for name, tensor in self._shadow.items():
                key = f"{_SHADOW_PREFIX}{name}"
                if key not in state:
                    raise KeyError(f"EMA state is missing {key!r}")
                _local(tensor).copy_(_local(state[key]).to(tensor.dtype))
        self._num_updates = int(state.get("num_updates", self._num_updates))
