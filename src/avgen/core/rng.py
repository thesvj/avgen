"""Purpose-separated random number streams.

A diffusion trainer draws randomness for four unrelated reasons: the noise added
to a sample, the timestep it is noised to, the conditioning dropout that makes
classifier-free guidance possible, and the sampler at evaluation time. Sharing
one generator between them couples those draws, so changing the number of
sampler steps silently changes the training noise and no two runs are
comparable.

:class:`RNGStreams` gives each purpose its own generator, derived from a single
seed by a fixed offset. Adding a fifth stream never perturbs the first four.

At scale, the second concern is *what varies per rank*. Data-parallel ranks must
draw different noise (otherwise the effective batch is one sample repeated), but
tensor-parallel and context-parallel ranks holding shards of the *same* sample
must draw identical noise. :func:`streams_for_rank` encodes that rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import torch

__all__ = ["RNGStreams"]

_MAX_SEED = 2**63 - 1


class RNGStreams:
    """Live generators separated by semantic purpose.

    Args:
        noise: Stream for the noise tensor added to clean latents.
        timestep: Stream for per-sample noise levels.
        conditioning: Stream for conditioning dropout and task sampling.
        sampler: Stream for inference-time sampling.
    """

    __slots__ = ("conditioning", "noise", "sampler", "timestep")

    #: Fixed odd offsets, chosen once and never changed: they are part of the
    #: reproducibility contract, so a run seeded 0 today matches one seeded 0 in
    #: a year. Derived seeds stay far apart in the 64-bit space.
    OFFSETS: ClassVar[dict[str, int]] = {
        "noise": 0x1D8E4E27C47D124F,
        "timestep": 0x4A39B70D6C2F8A11,
        "conditioning": 0x6F05C2E91B743D35,
        "sampler": 0x2C917AF5E30864DB,
    }

    def __init__(
        self,
        noise: torch.Generator,
        timestep: torch.Generator,
        conditioning: torch.Generator,
        sampler: torch.Generator,
    ) -> None:
        self.noise = noise
        self.timestep = timestep
        self.conditioning = conditioning
        self.sampler = sampler

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Self:
        """Create deterministic independent streams on one execution device.

        Args:
            seed: Non-negative base seed.
            device: Device the generators draw on. Generators are device bound
                in PyTorch, so this must match the device of the tensors being
                filled.

        Returns:
            Four independent streams.

        Raises:
            ValueError: If ``seed`` is negative or a bool.
        """
        if isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
        generators: dict[str, torch.Generator] = {}
        for name, offset in cls.OFFSETS.items():
            generator = torch.Generator(device=device)
            generator.manual_seed((seed + offset) % _MAX_SEED)
            generators[name] = generator
        return cls(**generators)

    @classmethod
    def for_rank(
        cls,
        seed: int,
        *,
        data_rank: int,
        device: torch.device | str = "cpu",
    ) -> Self:
        """Create streams that vary across data-parallel ranks only.

        Ranks that hold *different* samples must draw different noise, or the
        effective batch collapses. Ranks that hold *shards of the same* sample
        (tensor parallel, context parallel, pipeline stages) must draw identical
        noise, or the shards disagree about what sample they are denoising.

        Passing the data-parallel coordinate, and only that coordinate,
        satisfies both rules at once.

        Args:
            seed: Base seed shared by the whole job.
            data_rank: Index of this rank within the data-parallel dimension,
                usually ``ParallelDims.data_rank``.
            device: Device the generators draw on.

        Returns:
            Streams for this rank.

        Raises:
            ValueError: If ``data_rank`` is negative.
        """
        if isinstance(data_rank, bool) or data_rank < 0:
            raise ValueError(f"data_rank must be non-negative; got {data_rank!r}")
        # A multiplicative stride keeps consecutive data ranks far apart in
        # seed space; adding the rank would give rank r of seed s the same
        # streams as rank r-1 of seed s+1.
        return cls.from_seed(
            (seed + data_rank * 0x9E3779B97F4A7C15) % _MAX_SEED, device=device
        )

    def fork(self, purpose: str) -> torch.Generator:
        """Return a fresh generator derived from this stream set.

        Useful for a nested component (an augmentation, a bucket sampler) that
        needs its own reproducible randomness without disturbing the four named
        streams.

        Args:
            purpose: A stable label. The same label always derives the same
                generator from the same stream state.

        Returns:
            A new generator on the same device as the noise stream.
        """
        base = int(torch.randint(0, _MAX_SEED, (1,), generator=self.noise).item())
        mixed = (base ^ (hash(purpose) & _MAX_SEED)) % _MAX_SEED
        generator = torch.Generator(device=self.noise.device)
        generator.manual_seed(mixed)
        return generator

    def state_dict(self) -> dict[str, Any]:
        """Return cloned generator states keyed by semantic stream name."""
        return {name: getattr(self, name).get_state().clone() for name in self.OFFSETS}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore every stream, rejecting missing or unexpected names.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            ValueError: If the key set does not match the named streams.
            TypeError: If a state tensor is not a CPU uint8 tensor.
        """
        expected = set(self.OFFSETS)
        observed = set(state)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "RNG state keys must match named streams; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name in self.OFFSETS:
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"RNG state {name!r} must be a tensor; got {type(value)}"
                )
            if value.dtype is not torch.uint8 or value.device.type != "cpu":
                raise TypeError(
                    f"RNG state {name!r} must be a CPU uint8 tensor; "
                    f"got {value.dtype} on {value.device}"
                )
            getattr(self, name).set_state(value)

    def validate(self) -> None:
        """Require all generators to target the same device.

        Raises:
            ValueError: If the streams straddle two devices, which would make
                the noise draw and the timestep draw land on different GPUs.
        """
        devices = {str(getattr(self, name).device) for name in self.OFFSETS}
        if len(devices) != 1:
            raise ValueError(
                "all RNG streams must target the same device; got "
                + ", ".join(sorted(devices))
            )
