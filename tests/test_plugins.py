"""Third-party extensions load through entry points, in every registry.

README and CONTRIBUTING both promise that a model, metric, sampler or reward can
ship in its own package and be named in a config with no fork and no import in
the training script. That promise is only worth making if all four groups are
actually wired — three of them were not, and nothing failed, because a registry
that never looks for plugins looks exactly like a registry with no plugins
installed.

The plugins here are installed for real: a distribution is written into a
temporary directory and put on `sys.path`, so `importlib.metadata` discovers it
the way it would discover a `pip install`. A monkeypatched `entry_points` would
pass while the wiring was still absent.
"""

from __future__ import annotations

import re
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

GROUPS = ("avgen.models", "avgen.metrics", "avgen.samplers", "avgen.rewards")


def install_distribution(root: Path, name: str, entry_points: str, module: str) -> None:
    """Write an importable package plus the metadata that advertises it.

    Args:
        root: Directory placed on ``sys.path``.
        name: Distribution name.
        entry_points: Body of ``entry_points.txt``.
        module: Source of the package's ``__init__.py``.
    """
    package = root / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(textwrap.dedent(module), encoding="utf-8")
    dist = root / f"{name}-1.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    (dist / "entry_points.txt").write_text(
        textwrap.dedent(entry_points).lstrip(), encoding="utf-8"
    )


@pytest.fixture
def plugin_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Install plugins for one test, and leave no trace in any registry.

    Two pieces of global state have to be restored, and missing either one makes
    a *different* test fail later in the same worker — which is the worst kind of
    failure to debug:

    * the loader's per-group memo, or discovery is skipped and the test passes
      vacuously (tests install plugins after the first real lookup has happened);
    * every registry this test could write into, or a plugin reward stays visible
      to a test that counts the shipped rewards.
    """
    import avgen._plugins as plugins
    from avgen.eval.protocols import _REGISTRY as METRICS
    from avgen.infer.sampler import _SAMPLERS
    from avgen.models.registry import _CONFIGS, _MODELS
    from avgen.rl.rewards import _REWARDS

    registries: tuple[dict[str, object], ...] = (
        METRICS,
        _SAMPLERS,
        _REWARDS,
        _MODELS,
        _CONFIGS,
    )
    snapshots = [dict(registry) for registry in registries]

    monkeypatch.syspath_prepend(str(tmp_path))
    saved_memo = set(plugins._LOADED)
    plugins._LOADED.clear()
    try:
        yield tmp_path
    finally:
        plugins._LOADED.clear()
        plugins._LOADED.update(saved_memo)
        for registry, snapshot in zip(registries, snapshots, strict=True):
            registry.clear()
            registry.update(snapshot)
        for module in [n for n in sys.modules if n.startswith("avgen_plugin")]:
            del sys.modules[module]


class TestEveryRegistryLoadsPlugins:
    def test_a_metric_is_discovered(self, plugin_path: Path) -> None:
        from avgen.eval.protocols import build_metric, list_metrics

        install_distribution(
            plugin_path,
            "avgen_plugin_metric",
            """
            [avgen.metrics]
            plugin_metric = avgen_plugin_metric:PluginMetric
            """,
            '''
            """A metric that only needs to be constructible."""
            from __future__ import annotations

            import torch


            class PluginMetric:
                name = "plugin_metric"

                def update(self, **kwargs: object) -> None:
                    return None

                def compute(self) -> dict[str, float]:
                    return {"plugin_metric/value": 1.0}

                def reset(self) -> None:
                    return None

                def state(self) -> dict[str, tuple[float, int]]:
                    return {}

                def merge(self, state: object) -> None:
                    return None
            ''',
        )
        assert "plugin_metric" in list_metrics()
        assert build_metric("plugin_metric").compute() == {"plugin_metric/value": 1.0}

    def test_a_sampler_is_discovered(self, plugin_path: Path) -> None:
        from avgen.infer.sampler import build_sampler, list_samplers

        install_distribution(
            plugin_path,
            "avgen_plugin_sampler",
            """
            [avgen.samplers]
            plugin_sampler = avgen_plugin_sampler:make_sampler
            """,
            '''
            """A sampler factory, which is what the registry stores."""
            from __future__ import annotations


            class PluginSampler:
                def __init__(self, config: object) -> None:
                    self.config = config

                evaluations_per_step = 1
                is_stochastic = False

                def reset(self) -> None:
                    return None

                def step(self, model_output, x_t, sigma_t, sigma_next, **kwargs):
                    return x_t + (sigma_next - sigma_t) * model_output


            def make_sampler(config: object) -> PluginSampler:
                return PluginSampler(config)
            ''',
        )
        assert "plugin_sampler" in list_samplers()
        assert build_sampler("plugin_sampler").evaluations_per_step == 1

    def test_a_reward_is_discovered(self, plugin_path: Path) -> None:
        from avgen.rl.rewards import build_reward, list_rewards

        install_distribution(
            plugin_path,
            "avgen_plugin_reward",
            """
            [avgen.rewards]
            plugin_reward = avgen_plugin_reward:make_reward
            """,
            '''
            """A reward is the extension most likely to be private."""
            from __future__ import annotations

            import torch


            class PluginReward:
                def score(self, media, prompts):
                    return torch.zeros(len(prompts))


            def make_reward(**options: object) -> PluginReward:
                return PluginReward()
            ''',
        )
        assert "plugin_reward" in list_rewards()
        assert build_reward("plugin_reward").score(None, ["a"]).shape == (1,)

    def test_a_model_is_discovered(self, plugin_path: Path) -> None:
        from avgen.models.registry import build_model, list_models

        install_distribution(
            plugin_path,
            "avgen_plugin_model",
            """
            [avgen.models]
            plugin_model = avgen_plugin_model:PluginModel
            """,
            '''
            """A model plugin, registered under its entry-point name."""
            from __future__ import annotations

            from dataclasses import dataclass

            from torch import nn


            @dataclass(frozen=True)
            class PluginModelConfig:
                width: int = 8


            class PluginModel(nn.Module):
                config_class = PluginModelConfig

                def __init__(self, config: PluginModelConfig) -> None:
                    super().__init__()
                    self.blocks = nn.ModuleList([nn.Linear(config.width, config.width)])
            ''',
        )
        assert "plugin_model" in list_models()
        assert len(build_model("plugin_model", {"width": 4}).blocks) == 1


class TestPluginFailuresAreLoud:
    def test_an_unimportable_plugin_names_the_group_and_the_value(
        self, plugin_path: Path
    ) -> None:
        """A plugin that silently does not load looks like a typo in the name.

        Hours later, on a cluster, in a message that mentions neither.
        """
        from avgen.rl.rewards import list_rewards

        install_distribution(
            plugin_path,
            "avgen_plugin_broken",
            """
            [avgen.rewards]
            broken = avgen_plugin_broken:does_not_exist
            """,
            '"""Deliberately does not define the advertised attribute."""',
        )
        with pytest.raises(RuntimeError, match=re.escape("avgen.rewards")) as caught:
            list_rewards()
        message = str(caught.value)
        assert "broken" in message
        assert "avgen_plugin_broken:does_not_exist" in message

    def test_a_plugin_of_the_wrong_shape_is_rejected_at_load(
        self, plugin_path: Path
    ) -> None:
        """Rejected when it loads, not at first use, which is far later."""
        from avgen.infer.sampler import list_samplers

        install_distribution(
            plugin_path,
            "avgen_plugin_notcallable",
            """
            [avgen.samplers]
            not_callable = avgen_plugin_notcallable:NOT_CALLABLE
            """,
            "NOT_CALLABLE = 42",
        )
        with pytest.raises(RuntimeError, match="must resolve to"):
            list_samplers()


class TestDiscoveryIsDeterministic:
    def test_entry_points_load_in_name_order(self, plugin_path: Path) -> None:
        """Unordered iteration here is a cross-rank determinism hazard.

        Two ranks with identical installed packages must register in identical
        order, or any registry whose iteration order reaches a computation
        diverges between them — silently, and only at scale.
        """
        import avgen._plugins as plugins

        install_distribution(
            plugin_path,
            "avgen_plugin_order",
            """
            [avgen.ordering_probe]
            zebra = avgen_plugin_order:zebra
            alpha = avgen_plugin_order:alpha
            middle = avgen_plugin_order:middle
            """,
            "zebra = 'z'\nalpha = 'a'\nmiddle = 'm'",
        )
        registry: dict[str, object] = {}
        plugins.load_entry_points("avgen.ordering_probe", registry, kind="probe")
        assert list(registry) == ["alpha", "middle", "zebra"]

    def test_a_group_is_only_scanned_once(self, plugin_path: Path) -> None:
        """The loader runs on every lookup, so the memo is what keeps it cheap."""
        import avgen._plugins as plugins

        install_distribution(
            plugin_path,
            "avgen_plugin_once",
            """
            [avgen.once_probe]
            thing = avgen_plugin_once:thing
            """,
            "thing = 'x'",
        )
        first: dict[str, object] = {}
        plugins.load_entry_points("avgen.once_probe", first, kind="probe")
        assert first == {"thing": "x"}

        second: dict[str, object] = {}
        plugins.load_entry_points("avgen.once_probe", second, kind="probe")
        assert second == {}, "the second scan should be a no-op, not a rescan"


def test_every_documented_group_is_actually_wired() -> None:
    """README promises four groups. A registry that never looks finds nothing.

    This is the test that would have caught three missing wirings: each registry
    has to name its group as a module constant, and the shared loader has to be
    the thing that reads it.
    """
    from avgen.eval.protocols import METRIC_ENTRY_POINT_GROUP
    from avgen.infer.sampler import SAMPLER_ENTRY_POINT_GROUP
    from avgen.models.registry import MODEL_ENTRY_POINT_GROUP
    from avgen.rl.rewards import REWARD_ENTRY_POINT_GROUP

    declared = {
        MODEL_ENTRY_POINT_GROUP,
        METRIC_ENTRY_POINT_GROUP,
        SAMPLER_ENTRY_POINT_GROUP,
        REWARD_ENTRY_POINT_GROUP,
    }
    assert declared == set(GROUPS)
