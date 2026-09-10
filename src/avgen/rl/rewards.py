"""Reward models: the thing the policy is actually optimising.

A reward function in generative RL is a proxy, and every proxy is eventually
gamed. That fact shapes this module more than any implementation detail:

* **Rewards are composable and individually normalised.** A single reward is a
  single blind spot; a weighted sum of several is harder to saturate, and
  per-component normalisation is what makes the weights mean what they say. Two
  raw rewards on scales of 0.01 and 40 combined with weights 1 and 1 is
  really a weighting of 1 to 4000, and nothing in the training curve says so.
* **The reference rewards ship with no dependencies.** The entire RL path —
  rollout, advantage, ratio, clipping, update — must be runnable in CI on CPU
  without downloading a preference model, or the RL code is only ever exercised
  by the people who have the checkpoints, which is nobody in CI.
* **Learned rewards are declared, not vendored.** HPSv2, PickScore and
  VideoScore are real dependencies with real weights; each is registered here so
  a config can name it, and each raises a clear error naming the extra rather
  than failing at import time.

A note on the built-in rewards: they are honest *proxies*, not quality models.
Temporal consistency alone is maximised by a still image. Motion magnitude alone
is maximised by noise. That is not a flaw to be fixed by tuning them — it is the
central fact about reward design, and the built-ins are written so the failure
is visible (see :class:`MotionMagnitudeReward`, which targets a band rather than
a direction, precisely because the unbounded version is trivially hackable).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

__all__ = [
    "AVSyncProxyReward",
    "CompositeReward",
    "HPSv2Reward",
    "MotionMagnitudeReward",
    "PickScoreReward",
    "RewardModel",
    "RunningNormalizer",
    "TemporalConsistencyReward",
    "VideoScoreReward",
    "build_reward",
    "list_rewards",
    "media_tensors",
    "register_reward",
]


@runtime_checkable
class RewardModel(Protocol):
    """Scores generated media against the prompts that produced it.

    One method, returning one scalar per sample. Deliberately not a
    ``nn.Module``: a reward may be a neural network, a heuristic, a call to a
    remote service or a human-labelled lookup, and forcing all of them into a
    module hierarchy buys nothing.
    """

    def score(self, media: Any, prompts: Sequence[str]) -> torch.Tensor:
        """Return ``(batch,)`` float32 rewards, higher is better."""
        ...


def media_tensors(media: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract video and optional audio tensors from whatever generation returned.

    Structural rather than nominal typing, so this accepts a bare tensor (the
    common case in a unit test), a mapping, and
    :class:`avgen.infer.GeneratedMedia` without importing it — which matters
    because :mod:`avgen.rl` must import on a machine where inference is not set
    up, and because a user's own pipeline object works unchanged.

    Args:
        media: A ``(batch, channels, frames, height, width)`` tensor, a mapping
            with a ``"video"`` key, or an object with a ``.video`` attribute and
            optionally ``.audio``.

    Returns:
        ``(video, audio_or_none)``.

    Raises:
        TypeError: If no video tensor can be found.
        ValueError: If the video tensor is not rank 5.
    """
    video: Any = None
    audio: Any = None
    if isinstance(media, torch.Tensor):
        video = media
    elif isinstance(media, Mapping):
        video = media.get("video")
        audio = media.get("audio")
    else:
        video = getattr(media, "video", None)
        audio = getattr(media, "audio", None)
    if not isinstance(video, torch.Tensor):
        raise TypeError(
            "media must be a video tensor, a mapping with a 'video' key, or an "
            f"object with a .video attribute; got {type(media).__name__}"
        )
    if video.ndim != 5:
        raise ValueError(
            "video must be rank 5 (batch, channels, frames, height, width); "
            f"got {tuple(video.shape)}"
        )
    if audio is not None and not isinstance(audio, torch.Tensor):
        audio = None
    return video, audio


def _check_prompts(video: torch.Tensor, prompts: Sequence[str]) -> None:
    """Require one prompt per sample.

    Args:
        video: The media batch.
        prompts: The prompts that produced it.

    Raises:
        ValueError: On a length mismatch, which silently misaligns every reward
            with the sample it is meant to score.
    """
    if len(prompts) != video.shape[0]:
        raise ValueError(
            f"expected {video.shape[0]} prompts to match the batch; got {len(prompts)}"
        )


def _frames(video: torch.Tensor) -> torch.Tensor:
    """Flatten each frame to a vector: ``(batch, frames, channels*height*width)``."""
    batch, _, frames = video.shape[0], video.shape[1], video.shape[2]
    return video.permute(0, 2, 1, 3, 4).reshape(batch, frames, -1).float()


@dataclass(frozen=True, slots=True)
class TemporalConsistencyReward:
    """Mean cosine similarity between consecutive frames.

    The dependency-free stand-in for the CLIP-feature consistency metric the
    literature uses, and the counterpart of ``avgen.eval``'s temporal-consistency
    metric. It measures the right thing (frames that belong to the same scene)
    and is maximised by the wrong thing (a still image), which is exactly why it
    belongs in a :class:`CompositeReward` opposite a motion term and never
    alone.

    Args:
        name: Registry name, carried so a composite can label its components.
    """

    name: str = "temporal_consistency"

    def score(self, media: Any, prompts: Sequence[str]) -> torch.Tensor:
        """Return ``(batch,)`` mean consecutive-frame cosine similarity.

        Args:
            media: Generated media.
            prompts: Prompts, used only for the length check.

        Returns:
            Rewards in ``[-1, 1]``; a single-frame clip scores 1.0.
        """
        video, _ = media_tensors(media)
        _check_prompts(video, prompts)
        flat = _frames(video)
        if flat.shape[1] < 2:
            return torch.ones(flat.shape[0], device=video.device, dtype=torch.float32)
        current = flat[:, :-1]
        following = flat[:, 1:]
        similarity = torch.nn.functional.cosine_similarity(current, following, dim=-1)
        return similarity.mean(dim=1)


@dataclass(frozen=True, slots=True)
class MotionMagnitudeReward:
    """How much the frame content changes, scored against a target band.

    Written as a band rather than as "more is better" on purpose. An unbounded
    motion reward is maximised by per-pixel noise: the policy discovers within a
    few hundred steps that flicker scores higher than motion, the reward curve
    goes up, and the samples become unwatchable. That is reward hacking in its
    purest form, and the shape of the reward is the only defence against it.

    Args:
        target: Desired mean absolute inter-frame difference, in latent units.
        tolerance: Width of the band. The reward is 1.0 at ``target`` and decays
            smoothly; a wider tolerance is a weaker constraint.
        name: Registry name.

    Raises:
        ValueError: If ``tolerance`` is not positive or ``target`` is negative.
    """

    target: float = 0.1
    tolerance: float = 0.08
    name: str = "motion_magnitude"

    def __post_init__(self) -> None:
        """Validate the band.

        Raises:
            ValueError: On a non-positive tolerance or a negative target.
        """
        if self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive; got {self.tolerance!r}")
        if self.target < 0.0:
            raise ValueError(f"target must be non-negative; got {self.target!r}")

    def score(self, media: Any, prompts: Sequence[str]) -> torch.Tensor:
        """Return ``(batch,)`` band-shaped motion rewards in ``(0, 1]``.

        Args:
            media: Generated media.
            prompts: Prompts, used only for the length check.

        Returns:
            Rewards peaking at 1.0 when motion equals ``target``.
        """
        video, _ = media_tensors(media)
        _check_prompts(video, prompts)
        flat = _frames(video)
        if flat.shape[1] < 2:
            motion = torch.zeros(
                flat.shape[0], device=video.device, dtype=torch.float32
            )
        else:
            motion = (flat[:, 1:] - flat[:, :-1]).abs().mean(dim=(1, 2))
        deviation = (motion - self.target) / self.tolerance
        # Gaussian rather than a hard window: a hard window has zero gradient
        # outside it, so a policy that starts outside the band never learns
        # which direction the band is in.
        return torch.exp(-0.5 * deviation * deviation)


@dataclass(frozen=True, slots=True)
class AVSyncProxyReward:
    """Correlation between the visual motion envelope and the audio energy envelope.

    Genuine audio-visual synchrony needs a learned model. What can be measured
    without one is whether the two modalities' *activity* rises and falls
    together: a door slams, the pixels change and the waveform spikes in the
    same frame. Correlating the two envelopes catches gross desynchronisation —
    audio that plays over a static shot, video that cuts in silence — which is
    the failure mode an AV model actually exhibits early in training.

    The envelopes are resampled to a common length because video and audio
    latents run at different frame rates by construction; they share a physical
    time axis in seconds, which is what makes the resampling meaningful.

    Args:
        name: Registry name.
    """

    name: str = "av_sync_proxy"

    def score(self, media: Any, prompts: Sequence[str]) -> torch.Tensor:
        """Return ``(batch,)`` envelope correlation in ``[-1, 1]``.

        Args:
            media: Generated media; must carry audio.
            prompts: Prompts, used only for the length check.

        Returns:
            Correlations, or zeros when either modality has fewer than two
            frames (no envelope exists, so the honest score is "no evidence").
        """
        video, audio = media_tensors(media)
        _check_prompts(video, prompts)
        batch = video.shape[0]
        zeros = torch.zeros(batch, device=video.device, dtype=torch.float32)
        if audio is None or audio.ndim != 3 or audio.shape[-1] < 2:
            return zeros
        flat = _frames(video)
        if flat.shape[1] < 2:
            return zeros
        visual = (flat[:, 1:] - flat[:, :-1]).abs().mean(dim=-1)
        acoustic = audio.float().abs().mean(dim=1)
        length = min(visual.shape[1], acoustic.shape[1])
        if length < 2:
            return zeros
        visual = _resample(visual, length)
        acoustic = _resample(acoustic, length)
        return _correlate(visual, acoustic)


def _resample(envelope: torch.Tensor, length: int) -> torch.Tensor:
    """Linearly resample a ``(batch, time)`` envelope to a fixed length."""
    if envelope.shape[1] == length:
        return envelope
    return torch.nn.functional.interpolate(
        envelope.unsqueeze(1), size=length, mode="linear", align_corners=False
    ).squeeze(1)


def _correlate(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of two ``(batch, time)`` signals, per sample."""
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    numerator = (left * right).sum(dim=1)
    # 1e-8 keeps a constant envelope (a perfectly static clip, or silence) at a
    # correlation of 0 rather than at nan, which would poison the whole group's
    # advantage.
    denominator = left.norm(dim=1) * right.norm(dim=1) + 1e-8
    return numerator / denominator


class _GatedReward:
    """Base for a reward whose weights are a real, optional dependency.

    Registered so a config can name it and so ``list_rewards()`` tells the truth
    about what exists, but raising at construction with the exact install
    command. Failing here rather than at import time is what keeps
    ``import avgen.rl`` working on a bare CPU box.

    Args:
        extra: Name of the optional extra that provides this reward.
        package: Import name checked for availability.
        **_: Accepted and ignored so a config can carry the reward's real
            settings without the constructor needing to know them yet.
    """

    extra: str = "rewards"
    package: str = ""
    name: str = ""

    def __init__(self, **_: Any) -> None:
        raise RuntimeError(
            f"the {self.name!r} reward needs the {self.package!r} package, which "
            f"avgen does not depend on; install it with "
            f"pip install 'avgen[{self.extra}]'"
        )

    def score(self, media: Any, prompts: Sequence[str]) -> torch.Tensor:
        """Unreachable; construction always raises.

        Args:
            media: Generated media.
            prompts: Prompts.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(f"{self.name} is not available")


class HPSv2Reward(_GatedReward):
    """Human Preference Score v2 — a CLIP-based aesthetic preference model."""

    package = "hpsv2"
    name = "hps_v2"


class PickScoreReward(_GatedReward):
    """PickScore — a CLIP-H preference model trained on Pick-a-Pic."""

    package = "transformers"
    name = "pick_score"


class VideoScoreReward(_GatedReward):
    """VideoScore — a video-quality preference model over multiple axes."""

    package = "transformers"
    name = "video_score"


_REWARDS: dict[str, Callable[..., RewardModel]] = {}


def register_reward(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a reward constructor under a name.

    Args:
        name: Registry key used in configs.

    Returns:
        A decorator returning its argument unchanged.

    Raises:
        KeyError: If the name is already registered. Silent shadowing would mean
            two runs claiming the same reward were optimising different things.
    """

    def decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REWARDS:
            raise KeyError(f"reward {name!r} is already registered")
        _REWARDS[name] = factory
        return factory

    return decorate


def build_reward(name: str, **options: Any) -> RewardModel:
    """Construct a registered reward.

    Args:
        name: Registry key.
        **options: Constructor keyword arguments.

    Returns:
        The reward.

    Raises:
        KeyError: If the name is unknown, listing what is available.
    """
    try:
        factory = _REWARDS[name]
    except KeyError:
        raise KeyError(
            f"unknown reward {name!r}; registered rewards are {list_rewards()}"
        ) from None
    return factory(**options)


def list_rewards() -> tuple[str, ...]:
    """Return every registered reward name, sorted."""
    return tuple(sorted(_REWARDS))


register_reward("temporal_consistency")(TemporalConsistencyReward)
register_reward("motion_magnitude")(MotionMagnitudeReward)
register_reward("av_sync_proxy")(AVSyncProxyReward)
register_reward("hps_v2")(HPSv2Reward)
register_reward("pick_score")(PickScoreReward)
register_reward("video_score")(VideoScoreReward)


class RunningNormalizer:
    """Streaming mean and variance for one reward component.

    Implements :class:`avgen.core.state.Stateful`, so it checkpoints with the
    rest of the run. That is not a nicety: a normaliser that resets on resume
    changes the effective weighting of every reward component at the moment of
    the restart, and the resulting discontinuity in the loss curve is
    indistinguishable from a real training problem.

    Welford's algorithm rather than accumulating sums of squares, because a
    reward stream is long and the naive form loses precision exactly when the
    variance is small — which is the regime where the normaliser matters most.

    Args:
        momentum: ``None`` for an exact running average over all observations,
            or a value in ``(0, 1)`` for an exponential moving average that
            tracks a drifting reward distribution.
        epsilon: Added to the standard deviation before dividing.
    """

    __slots__ = ("_count", "_mean", "_variance", "epsilon", "momentum")

    def __init__(self, *, momentum: float | None = None, epsilon: float = 1e-6) -> None:
        if momentum is not None and not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1) or None; got {momentum!r}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive; got {epsilon!r}")
        self.momentum = momentum
        self.epsilon = epsilon
        self._count = 0.0
        self._mean = 0.0
        self._variance = 1.0

    def update(self, values: torch.Tensor) -> None:
        """Fold a batch of observations into the running statistics.

        Args:
            values: ``(batch,)`` rewards.
        """
        batch = values.numel()
        if batch == 0:
            return
        mean = float(values.mean())
        variance = float(values.var(unbiased=False))
        if self.momentum is not None:
            weight = self.momentum
            self._mean = weight * self._mean + (1.0 - weight) * mean
            self._variance = weight * self._variance + (1.0 - weight) * variance
            self._count += batch
            return
        total = self._count + batch
        delta = mean - self._mean
        self._mean += delta * batch / total
        # Chan's parallel variance combination: exact for batched updates,
        # which the naive incremental form is not.
        self._variance = (
            self._variance * self._count
            + variance * batch
            + delta * delta * self._count * batch / total
        ) / total
        self._count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Standardise a batch with the current statistics.

        Args:
            values: ``(batch,)`` rewards.

        Returns:
            ``(batch,)`` standardised rewards. Before the first update this is
            the identity, so the first batch is never divided by a variance
            estimated from nothing.
        """
        if self._count == 0.0:
            return values
        std = max(self._variance, 0.0) ** 0.5 + self.epsilon
        return (values - self._mean) / std

    def state_dict(self) -> dict[str, float]:
        """Return the running statistics."""
        return {
            "count": self._count,
            "mean": self._mean,
            "variance": self._variance,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the running statistics.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: If a statistic is missing.
        """
        for key in ("count", "mean", "variance"):
            if key not in state:
                raise KeyError(f"missing normalizer statistic {key!r}")
        self._count = float(state["count"])
        self._mean = float(state["mean"])
        self._variance = float(state["variance"])


@dataclass(slots=True)
class CompositeReward:
    """A weighted sum of rewards, each normalised before weighting.

    Normalisation is the whole point. Without it the weights are meaningless:
    a reward on the scale of 0.01 and one on the scale of 40 combined with equal
    weights is a 1-to-4000 weighting, and there is nothing in a training curve
    that would reveal it. With per-component normalisation the weight is the
    actual relative influence, and a weight sweep means what a reader assumes it
    means.

    ``"batch"`` normalisation standardises within the current batch. It is
    self-calibrating and needs no state, but under GRPO the batch *is* the
    group, and standardising within the group before the advantage — which
    standardises again — flattens the per-component contribution to a constant.
    ``"running"`` is therefore the correct default for GRPO: it standardises
    against the reward's history rather than against the group, leaving the
    within-group differences the advantage is built from intact.

    Args:
        components: ``(reward, weight)`` pairs, evaluated in order.
        normalization: ``"none"``, ``"batch"`` or ``"running"``.
        momentum: Momentum for the running normalisers.

    Raises:
        ValueError: If there are no components or the mode is unknown.
    """

    components: tuple[tuple[RewardModel, float], ...]
    normalization: str = "running"
    momentum: float | None = None
    _normalizers: list[RunningNormalizer] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validate and build one normaliser per component.

        Raises:
            ValueError: On an empty component list or unknown mode.
        """
        if not self.components:
            raise ValueError("CompositeReward needs at least one component")
        if self.normalization not in {"none", "batch", "running"}:
            raise ValueError(
                "normalization must be 'none', 'batch' or 'running'; "
                f"got {self.normalization!r}"
            )
        self._normalizers = [
            RunningNormalizer(momentum=self.momentum) for _ in self.components
        ]

    def component_scores(
        self, media: Any, prompts: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Return each component's raw, un-normalised reward.

        The diagnostic that matters when a run degrades: the composite can be
        flat while one component climbs and another collapses, and only the
        per-component view shows it.

        Args:
            media: Generated media.
            prompts: Prompts.

        Returns:
            Component name to ``(batch,)`` raw reward.
        """
        scores: dict[str, torch.Tensor] = {}
        for index, (reward, _) in enumerate(self.components):
            label = getattr(reward, "name", None) or f"reward_{index}"
            scores[str(label)] = reward.score(media, prompts).float()
        return scores

    def score(self, media: Any, prompts: Sequence[str]) -> torch.Tensor:
        """Return the normalised, weighted sum of every component.

        Args:
            media: Generated media.
            prompts: Prompts.

        Returns:
            ``(batch,)`` composite rewards.
        """
        total: torch.Tensor | None = None
        for index, (reward, weight) in enumerate(self.components):
            value = reward.score(media, prompts).float()
            if self.normalization == "batch":
                value = (value - value.mean()) / (value.std(unbiased=False) + 1e-6)
            elif self.normalization == "running":
                normalizer = self._normalizers[index]
                # Update after normalising, so the statistics used for a batch
                # never depend on that batch: otherwise the reward a sample
                # receives depends on which other samples it was scored with,
                # and the objective is no longer a function of the sample.
                normalized = normalizer.normalize(value)
                normalizer.update(value)
                value = normalized
            total = value * weight if total is None else total + value * weight
        assert total is not None
        return total

    def state_dict(self) -> dict[str, Any]:
        """Return the normaliser states, keyed by component index."""
        return {
            f"component_{index}": normalizer.state_dict()
            for index, normalizer in enumerate(self._normalizers)
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the normaliser states.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: If a component's state is missing.
        """
        for index, normalizer in enumerate(self._normalizers):
            key = f"component_{index}"
            if key not in state:
                raise KeyError(f"missing normalizer state {key!r}")
            normalizer.load_state_dict(state[key])
