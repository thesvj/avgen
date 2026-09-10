"""Aspect-ratio and duration bucketing, with a progressive curriculum.

**Why bucketing is not optional.** A real video corpus is a mess of aspect
ratios and durations. There are three ways to train a transformer on that:

1. **Resize everything to one shape.** Cheap, and it teaches the model that the
   world is 16:9 — or that faces are ovals, if the resize is anisotropic.
   Whatever the corpus's real distribution of framings was, it is gone.
2. **Pad every clip to the largest shape and mask.** Correct, and unaffordable.
   Attention is quadratic in sequence length, so padding a 256x256 clip up to
   1024x1024 costs sixteen times the tokens and two hundred and fifty six times
   the attention work, essentially all of it on masked positions. On a mixed
   corpus this is most of your compute budget spent on nothing.
3. **Bucket.** Group samples into a small set of (duration, resolution) classes,
   and build every batch from a single class. Cost is proportional to real
   content, no clip is squashed, and ``torch.compile`` sees a handful of static
   shapes rather than one per clip.

Only the third scales. Everything in this module exists to make it
deterministic and resumable.

**Why a curriculum.** Attention cost grows quadratically with sequence length,
so early training — where the model is learning texture, colour, and gross
motion, none of which need long context — is far cheaper at low resolution and
short duration. Shifting the bucket mixture toward the expensive buckets as
training progresses buys a large fraction of the total budget back. The mixture
is a *weight vector over buckets that varies with step*, interpolated between
keyframes, which keeps the schedule inspectable and reproducible instead of
hiding it in a loop of ``if step > n``.

**Why largest remainder.** Turning fractional mixture weights into an integer
number of batches per bucket is an apportionment problem, and naive rounding
does not sum to the target: round-half-up on five buckets weighted 0.1 each
gives zero batches. Largest remainder (Hamilton's method) allocates the floors,
then hands the leftover batches to the buckets with the largest fractional
parts. It is exact, it is deterministic, and its bias is bounded by one batch
per bucket.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from avgen.core._validate import require_dimension, require_positive
from avgen.data.protocols import BucketPlan

__all__ = [
    "BucketBatch",
    "BucketCurriculum",
    "BucketSampler",
    "CurriculumPhase",
    "largest_remainder",
]

#: Mixed into the epoch seed so bucket shuffling is decorrelated from any other
#: consumer of the same base seed. Two streams derived by adding small integers
#: to one seed are not independent, and the correlation shows up as the noise
#: draw tracking the sample order.
_BUCKET_SEED_SALT = 0x5D9E_3B71_4C08_A2F3
_MAX_SEED = (1 << 63) - 1


def largest_remainder(total: int, weights: Sequence[float]) -> tuple[int, ...]:
    """Apportion an integer total across weights so the parts sum exactly.

    Args:
        total: Number of indivisible units to hand out.
        weights: Non-negative weights. Need not sum to one.

    Returns:
        One integer per weight, summing exactly to ``total``.

    Raises:
        ValueError: If ``total`` is negative, ``weights`` is empty, a weight is
            negative or non-finite, or every weight is zero.
    """
    require_dimension("total", total, allow_zero=True)
    if not weights:
        raise ValueError("largest_remainder requires at least one weight")
    for index, weight in enumerate(weights):
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"weights[{index}] must be finite and non-negative; got {weight!r}"
            )
    mass = math.fsum(weights)
    if mass <= 0.0:
        raise ValueError("largest_remainder requires at least one positive weight")

    exact = [total * weight / mass for weight in weights]
    floors = [int(math.floor(value)) for value in exact]
    leftover = total - sum(floors)
    # Ties break on index, never on a set or dict ordering: the apportionment
    # must be identical on every rank, and two ranks that disagree about which
    # bucket got the spare batch produce different batch counts and deadlock at
    # the next collective.
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - floors[index]), index),
    )
    for position in range(leftover):
        floors[order[position % len(order)]] += 1
    return tuple(floors)


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    """One keyframe of a bucket mixture schedule.

    Args:
        start_step: Optimizer step at which this keyframe's weights are exact.
        weights: One non-negative weight per bucket, in plan order. Normalised
            when the curriculum is evaluated, so counts or fractions both work.

    Raises:
        ValueError: If the step is negative, the weight vector is empty, a
            weight is negative or non-finite, or every weight is zero.
    """

    start_step: int
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the step and the weight vector."""
        require_dimension("start_step", self.start_step, allow_zero=True)
        if not self.weights:
            raise ValueError("a curriculum phase requires at least one weight")
        for index, weight in enumerate(self.weights):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"weights[{index}] must be finite and non-negative; "
                    f"got {weight!r}"
                )
        if math.fsum(self.weights) <= 0.0:
            raise ValueError("a curriculum phase must have one positive weight")


@dataclass(frozen=True, slots=True)
class BucketCurriculum:
    """A bucket mixture that varies with the optimizer step.

    Between keyframes the weights are linearly interpolated by default. The
    rejected alternative was step functions: a discontinuous jump from "all
    short clips" to "all long clips" is a distribution shift the optimizer
    experiences as a loss spike, and at large batch sizes that spike is
    sometimes unrecoverable. Ramping keeps every bucket in the mixture
    throughout, which also means the model never fully forgets a resolution it
    saw earlier.

    Args:
        phases: Keyframes, which are sorted by step on construction.
        interpolate: Whether to ramp between keyframes. False makes each
            keyframe hold until the next, which is occasionally what an ablation
            wants.

    Raises:
        ValueError: If there are no phases, two phases share a step, or the
            phases disagree about how many buckets there are.
    """

    phases: tuple[CurriculumPhase, ...]
    interpolate: bool = True

    def __post_init__(self) -> None:
        """Sort the keyframes and validate their consistency."""
        if not self.phases:
            raise ValueError("BucketCurriculum requires at least one phase")
        widths = {len(phase.weights) for phase in self.phases}
        if len(widths) != 1:
            raise ValueError(
                "every curriculum phase must carry one weight per bucket; got "
                f"vectors of lengths {sorted(widths)}"
            )
        steps = [phase.start_step for phase in self.phases]
        if len(set(steps)) != len(steps):
            raise ValueError(f"curriculum phases must have distinct steps; got {steps}")
        object.__setattr__(
            self,
            "phases",
            tuple(sorted(self.phases, key=lambda phase: phase.start_step)),
        )

    @property
    def bucket_count(self) -> int:
        """Number of buckets the weight vectors describe."""
        return len(self.phases[0].weights)

    def weights_at(self, step: int) -> tuple[float, ...]:
        """Return the normalised mixture weights at a training step.

        Steps before the first keyframe use the first keyframe's weights, and
        steps after the last use the last keyframe's. Clamping rather than
        extrapolating matters: a linear extrapolation past the final keyframe
        goes negative, and a negative mixture weight is not a thing.

        Args:
            step: Optimizer step.

        Returns:
            Weights summing to one, in plan order.

        Raises:
            ValueError: If ``step`` is negative.
        """
        require_dimension("step", step, allow_zero=True)
        if step <= self.phases[0].start_step:
            return _normalise(self.phases[0].weights)
        if step >= self.phases[-1].start_step:
            return _normalise(self.phases[-1].weights)
        for left, right in zip(self.phases, self.phases[1:], strict=False):
            if left.start_step <= step < right.start_step:
                if not self.interpolate:
                    return _normalise(left.weights)
                span = right.start_step - left.start_step
                alpha = (step - left.start_step) / span
                blended = tuple(
                    (1.0 - alpha) * low + alpha * high
                    for low, high in zip(left.weights, right.weights, strict=True)
                )
                return _normalise(blended)
        return _normalise(self.phases[-1].weights)


def _normalise(weights: Sequence[float]) -> tuple[float, ...]:
    """Scale weights to sum to one.

    Args:
        weights: Non-negative weights with a positive sum.

    Returns:
        The normalised weights.
    """
    mass = math.fsum(weights)
    return tuple(weight / mass for weight in weights)


@dataclass(frozen=True, slots=True)
class BucketBatch:
    """One batch's worth of sample indices, all from the same bucket.

    Args:
        bucket_id: The bucket every sample belongs to.
        indices: Global sample indices, in the order they should be collated.
    """

    bucket_id: int
    indices: tuple[int, ...]

    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        return len(self.indices)


class BucketSampler:
    """Deterministic, curriculum-aware batch planning over a bucketed corpus.

    The sampler is a pure function of ``(seed, epoch, step, assignments)``. It
    holds no iteration state, and it is not an iterator: it *plans* an epoch,
    returning the complete batch list. That is what makes resumption exact — a
    loader restores its position by replaying the plan and skipping forward,
    with no need to serialise a generator's internals — and it is what lets
    every rank in a job compute the same plan independently instead of
    broadcasting it.

    Args:
        plan: The buckets and their base mixture.
        assignments: One bucket id per corpus sample, in global index order.
        batch_size: Samples per batch. Every emitted batch is exactly this size.
        seed: Base seed for shuffling.
        curriculum: Optional step-varying mixture. When absent the plan's own
            weights are used at every step.
        allow_oversampling: Whether a bucket may be drawn from more times than
            it has samples in one epoch. True is the useful default under a
            curriculum: the mixture deliberately over-weights rare expensive
            buckets, and refusing to repeat their samples would silently cap the
            mixture at the corpus census. False makes a quota larger than the
            pool an error.

    Raises:
        ValueError: If the corpus is empty, an assignment names an unknown
            bucket, or a bucket holds fewer samples than one batch.
    """

    __slots__ = (
        "_allow_oversampling",
        "_batch_size",
        "_curriculum",
        "_plan",
        "_pools",
        "_seed",
    )

    def __init__(
        self,
        plan: BucketPlan,
        assignments: Sequence[int],
        *,
        batch_size: int,
        seed: int = 0,
        curriculum: BucketCurriculum | None = None,
        allow_oversampling: bool = True,
    ) -> None:
        require_positive("batch_size", batch_size)
        require_dimension("seed", seed, allow_zero=True)
        if not assignments:
            raise ValueError("BucketSampler requires a non-empty corpus")
        if curriculum is not None and curriculum.bucket_count != len(plan):
            raise ValueError(
                f"curriculum describes {curriculum.bucket_count} buckets but the "
                f"plan has {len(plan)}"
            )
        pools: list[list[int]] = [[] for _ in plan]
        for index, bucket_id in enumerate(assignments):
            pools[plan.index_of(bucket_id)].append(index)
        for position, pool in enumerate(pools):
            if 0 < len(pool) < batch_size:
                raise ValueError(
                    f"bucket {plan.buckets[position].bucket_id} holds {len(pool)} "
                    f"samples, fewer than batch_size={batch_size}; drop the bucket "
                    "or lower the batch size rather than emitting a ragged batch "
                    "that would force a recompile"
                )
        self._plan = plan
        self._pools = tuple(tuple(pool) for pool in pools)
        self._batch_size = batch_size
        self._seed = seed
        self._curriculum = curriculum
        self._allow_oversampling = allow_oversampling

    @property
    def plan(self) -> BucketPlan:
        """The bucket plan this sampler draws from."""
        return self._plan

    @property
    def batch_size(self) -> int:
        """Samples per emitted batch."""
        return self._batch_size

    def pool_size(self, bucket_id: int) -> int:
        """Return how many corpus samples fall into a bucket.

        Args:
            bucket_id: Bucket to query.

        Returns:
            The number of samples assigned to it.

        Raises:
            KeyError: If the bucket is not in the plan.
        """
        return len(self._pools[self._plan.index_of(bucket_id)])

    def batches_per_epoch(self) -> int:
        """Return how many batches one pass over the corpus produces.

        This is the epoch length under *any* mixture, not just the current one:
        the curriculum redistributes batches between buckets but never changes
        how many there are. Keeping the epoch length independent of the mixture
        is what stops the learning-rate schedule from stretching or compressing
        when the curriculum moves.
        """
        return sum(len(pool) // self._batch_size for pool in self._pools)

    def weights_at(self, step: int) -> tuple[float, ...]:
        """Return the mixture weights in force at a step.

        Buckets with no samples are zeroed and the remainder renormalised,
        because asking for batches from an empty bucket cannot be satisfied by
        any amount of oversampling.

        Args:
            step: Optimizer step.

        Returns:
            Weights summing to one, in plan order.

        Raises:
            ValueError: If every non-empty bucket has zero weight.
        """
        base = (
            self._curriculum.weights_at(step)
            if self._curriculum is not None
            else self._plan.weights
        )
        masked = tuple(
            weight if pool else 0.0
            for weight, pool in zip(base, self._pools, strict=True)
        )
        if math.fsum(masked) <= 0.0:
            raise ValueError(
                f"at step {step} every bucket with a positive weight is empty; "
                "the curriculum asks for data the corpus does not contain"
            )
        return _normalise(masked)

    def quotas_at(self, step: int, *, batches: int | None = None) -> tuple[int, ...]:
        """Return the integer batch count per bucket at a step.

        Args:
            step: Optimizer step.
            batches: Total batches to apportion. Defaults to
                :meth:`batches_per_epoch`.

        Returns:
            One count per bucket, in plan order, summing to ``batches``.

        Raises:
            ValueError: If a quota exceeds a bucket's pool and oversampling is
                disallowed.
        """
        total = self.batches_per_epoch() if batches is None else batches
        quotas = largest_remainder(total, self.weights_at(step))
        if not self._allow_oversampling:
            for position, quota in enumerate(quotas):
                available = len(self._pools[position]) // self._batch_size
                if quota > available:
                    bucket = self._plan.buckets[position]
                    raise ValueError(
                        f"bucket {bucket.bucket_id} is asked for {quota} batches at "
                        f"step {step} but only holds {available}; enable "
                        "allow_oversampling or soften the curriculum"
                    )
        return quotas

    def plan_epoch(self, *, epoch: int, step: int = 0) -> tuple[BucketBatch, ...]:
        """Return the complete, ordered batch plan for one epoch.

        The construction, in order, and every step of it seeded from
        ``(seed, epoch, ...)`` alone so that any rank can reproduce it:

        1. Apportion the epoch's batches across buckets by largest remainder.
        2. Shuffle each bucket's pool independently.
        3. Take the required samples from each pool, wrapping around with a
           fresh shuffle when a quota exceeds the pool. Re-shuffling on the wrap
           rather than repeating the same order means an oversampled bucket
           pairs its samples differently on the second pass, which is a weaker
           form of augmentation but a real one.
        4. Shuffle the *order of batches* so buckets interleave. Without this
           the model sees all of bucket zero, then all of bucket one: a
           resolution schedule nobody asked for, and one that interacts badly
           with momentum.

        Args:
            epoch: Pass number over the corpus. Part of the shuffle seed, so
                every epoch is a different permutation.
            step: Optimizer step at the start of the epoch, used to evaluate the
                curriculum. Evaluated once per epoch rather than per batch so
                that the mixture is a property of the epoch plan and a resumed
                epoch replays identically.

        Returns:
            The epoch's batches, in emission order.

        Raises:
            ValueError: If ``epoch`` is negative, or a quota cannot be met.
        """
        require_dimension("epoch", epoch, allow_zero=True)
        quotas = self.quotas_at(step)
        batches: list[BucketBatch] = []
        for position, quota in enumerate(quotas):
            if quota == 0:
                continue
            pool = self._pools[position]
            bucket_id = self._plan.buckets[position].bucket_id
            needed = quota * self._batch_size
            drawn = self._draw(pool, needed, epoch=epoch, bucket_id=bucket_id)
            for start in range(0, needed, self._batch_size):
                batches.append(
                    BucketBatch(
                        bucket_id=bucket_id,
                        indices=tuple(drawn[start : start + self._batch_size]),
                    )
                )
        order = _permutation(
            len(batches), seed=_epoch_seed(self._seed, epoch, salt=0xB47C)
        )
        return tuple(batches[position] for position in order)

    def _draw(
        self, pool: Sequence[int], count: int, *, epoch: int, bucket_id: int
    ) -> list[int]:
        """Draw ``count`` indices from a bucket pool, wrapping if necessary.

        Args:
            pool: Global sample indices in this bucket.
            count: How many to draw.
            epoch: Epoch number, part of the shuffle seed.
            bucket_id: Bucket identifier, part of the shuffle seed so two
                buckets never receive correlated permutations.

        Returns:
            The drawn indices.

        Raises:
            ValueError: If the pool is empty.
        """
        if not pool:
            raise ValueError(f"bucket {bucket_id} has no samples to draw from")
        drawn: list[int] = []
        pass_index = 0
        while len(drawn) < count:
            seed = _epoch_seed(self._seed, epoch, salt=bucket_id * 0x9E37 + pass_index)
            order = _permutation(len(pool), seed=seed)
            drawn.extend(pool[position] for position in order)
            pass_index += 1
        return drawn[:count]


def _epoch_seed(base_seed: int, epoch: int, *, salt: int) -> int:
    """Derive a shuffle seed from a base seed, an epoch, and a purpose salt.

    Args:
        base_seed: The job's base seed.
        epoch: Pass number over the corpus.
        salt: Purpose discriminator.

    Returns:
        A seed in the range accepted by ``torch.Generator.manual_seed``.
    """
    mixed = (base_seed * 0x9E37_79B9_7F4A_7C15) ^ (epoch * 0xC2B2_AE3D_27D4_EB4F)
    mixed ^= (salt + _BUCKET_SEED_SALT) * 0x1656_67B1_9E37_79F9
    return (mixed & ((1 << 64) - 1)) % _MAX_SEED


def _permutation(count: int, *, seed: int) -> tuple[int, ...]:
    """Return a deterministic permutation of ``range(count)``.

    ``torch.randperm`` with an explicit generator is used rather than
    ``random.shuffle`` because torch's generator is the one whose algorithm is
    pinned across releases and platforms; Python's Mersenne Twister shuffle is
    stable in practice but is not part of any compatibility guarantee.

    Args:
        count: Length of the permutation.
        seed: Seed for the permutation.

    Returns:
        The permutation.
    """
    if count == 0:
        return ()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return tuple(
        int(value) for value in torch.randperm(count, generator=generator).tolist()
    )
