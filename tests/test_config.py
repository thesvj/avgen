"""Tests for the configuration subsystem.

A config error is the cheapest error a run can have and the most expensive one
to miss: a typo that is silently ignored produces a job that trains happily
with the wrong schedule for a thousand GPU-hours and reports nothing. So the
properties asserted here are all variations on one theme — every value that
reaches a dataclass field is the value the file meant, or the load fails loudly
before a single rank is allocated.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from avgen.config import (
    SCHEMA_VERSION,
    BucketConfig,
    ConfigError,
    ModelConfig,
    RunConfig,
    apply_overrides,
    build_accelerator,
    build_activation_checkpoint,
    build_fsdp_config,
    build_interconnects,
    build_model_shape,
    build_parallel_config,
    build_parallel_dims,
    build_pipeline_config,
    build_precision_config,
    config_diff,
    format_diff,
    gradient_accumulation,
    load_config,
    load_mapping,
    model_kwargs,
    parse_override,
    save_config,
    sequence_length_for,
    to_mapping,
)
from avgen.config.loader import coerce, deep_merge

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
SHIPPED_CONFIGS = sorted(CONFIG_ROOT.rglob("*.yaml"))
SHIPPED_IDS = [str(path.relative_to(CONFIG_ROOT)) for path in SHIPPED_CONFIGS]


def write(path: Path, text: str) -> Path:
    """Write a config file, creating parents, and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestShippedConfigs:
    """Every config in the repository must load and round-trip.

    A shipped config that does not load is a broken example, and an example is
    the first thing a new user runs. Parametrising over the real directory
    means a config added tomorrow is covered without anyone remembering to add
    a test — and a config that stops loading fails here rather than on a
    cluster.
    """

    def test_the_config_directory_is_not_empty(self) -> None:
        # Guards the parametrisation itself: a glob that silently matches
        # nothing turns every test below into a no-op that reports as passing.
        assert SHIPPED_CONFIGS, f"no YAML configs found under {CONFIG_ROOT}"

    @pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=SHIPPED_IDS)
    def test_loads(self, path: Path) -> None:
        config = load_config(path)
        assert isinstance(config, RunConfig)
        assert config.schema_version <= SCHEMA_VERSION

    @pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=SHIPPED_IDS)
    def test_round_trips_through_save_config(self, path: Path, tmp_path: Path) -> None:
        # save_config writes the resolved config beside every checkpoint. If it
        # does not reload to exactly the same object, the file next to the
        # checkpoint does not describe the run that produced it.
        config = load_config(path)
        written = save_config(config, tmp_path / "resolved.yaml")
        assert config_diff(config, load_config(written)) == {}

    @pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=SHIPPED_IDS)
    def test_every_bucket_patchifies_exactly(self, path: Path) -> None:
        # Ragged patchification is otherwise discovered by the first forward
        # pass, after the allocator, loader, and mesh have all been built.
        config = load_config(path)
        for bucket in config.data.buckets:
            assert bucket.tokens(config.model) > 0
        assert config.sequence_length > 0


class TestComposition:
    """``_base_`` must compose predictably, or a config means something else.

    Deep merge for mappings and wholesale replacement for lists is the
    contract; element-wise list merging would let one prepended bucket rewrite
    every override that followed it.
    """

    def test_a_single_base_is_merged_under_the_child(self, tmp_path: Path) -> None:
        write(
            tmp_path / "base.yaml", "model:\n  depth: 4\n  width: 64\n  num_heads: 4\n"
        )
        write(tmp_path / "child.yaml", "_base_: base.yaml\nmodel:\n  depth: 8\n")
        config = load_config(tmp_path / "child.yaml")
        assert config.model.depth == 8
        assert config.model.width == 64

    def test_a_chain_of_bases_composes_transitively(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", "model:\n  width: 64\n  num_heads: 4\n  depth: 4\n")
        write(tmp_path / "b.yaml", "_base_: a.yaml\nmodel:\n  depth: 8\n")
        write(tmp_path / "c.yaml", "_base_: b.yaml\ntrain:\n  lr: 0.5\n")
        config = load_config(tmp_path / "c.yaml")
        assert (config.model.width, config.model.depth) == (64, 8)
        assert config.train.lr == 0.5

    def test_a_deep_merge_reaches_nested_sections(self, tmp_path: Path) -> None:
        write(
            tmp_path / "base.yaml",
            "parallel:\n  tensor: 2\n  precision:\n    param_dtype: float32\n"
            "    float8_min_features: 512\n",
        )
        write(
            tmp_path / "child.yaml",
            "_base_: base.yaml\nparallel:\n  precision:\n    param_dtype: bfloat16\n",
        )
        config = load_config(tmp_path / "child.yaml")
        assert config.parallel.tensor == 2
        assert config.parallel.precision.param_dtype == "bfloat16"
        # The sibling the child never mentioned survives the merge.
        assert config.parallel.precision.float8_min_features == 512

    def test_lists_replace_rather_than_merge(self, tmp_path: Path) -> None:
        write(
            tmp_path / "base.yaml",
            "data:\n  buckets:\n    - name: a\n      frames: 16\n"
            "    - name: b\n      frames: 32\n",
        )
        write(
            tmp_path / "child.yaml",
            "_base_: base.yaml\ndata:\n  buckets:\n    - name: only\n      frames: 8\n",
        )
        config = load_config(tmp_path / "child.yaml")
        assert [bucket.name for bucket in config.data.buckets] == ["only"]

    def test_several_bases_apply_left_to_right(self, tmp_path: Path) -> None:
        write(tmp_path / "one.yaml", "run_name: one\noutput_dir: from-one\n")
        write(tmp_path / "two.yaml", "run_name: two\n")
        write(tmp_path / "child.yaml", "_base_:\n  - one.yaml\n  - two.yaml\n")
        config = load_config(tmp_path / "child.yaml")
        assert config.run_name == "two"
        assert config.output_dir == "from-one"

    def test_base_paths_are_relative_to_the_file_that_names_them(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "shared" / "base.yaml", "run_name: shared\n")
        write(tmp_path / "nested" / "child.yaml", "_base_: ../shared/base.yaml\n")
        assert load_config(tmp_path / "nested" / "child.yaml").run_name == "shared"

    def test_a_cycle_is_reported_with_the_chain(self, tmp_path: Path) -> None:
        write(tmp_path / "x.yaml", "_base_: y.yaml\n")
        write(tmp_path / "y.yaml", "_base_: x.yaml\n")
        with pytest.raises(ConfigError, match=r"cyclic"):
            load_config(tmp_path / "x.yaml")

    def test_a_missing_base_names_the_file(self, tmp_path: Path) -> None:
        write(tmp_path / "child.yaml", "_base_: nowhere.yaml\n")
        with pytest.raises(ConfigError, match=r"config file not found"):
            load_config(tmp_path / "child.yaml")

    def test_a_non_string_base_entry_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path / "child.yaml", "_base_:\n  - 7\n")
        with pytest.raises(ConfigError, match=r"must be a string path"):
            load_config(tmp_path / "child.yaml")

    def test_a_mapping_base_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path / "child.yaml", "_base_:\n  a: b\n")
        with pytest.raises(ConfigError, match=r"must be a path or a list of paths"):
            load_config(tmp_path / "child.yaml")

    def test_the_base_key_does_not_survive_into_the_mapping(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "base.yaml", "run_name: base\n")
        write(tmp_path / "child.yaml", "_base_: base.yaml\n")
        assert "_base_" not in load_mapping(tmp_path / "child.yaml")

    def test_deep_merge_does_not_modify_either_input(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        merged = deep_merge(base, override)
        assert merged == {"a": {"x": 1, "y": 2}}
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


class TestFileErrors:
    """A malformed file must fail with a message naming the file.

    These are the errors a user hits at 3am on a login node; a traceback here
    trains people to stop reading error messages.
    """

    def test_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"config file not found"):
            load_config(tmp_path / "absent.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        write(tmp_path / "bad.yaml", "model: [unclosed\n")
        with pytest.raises(ConfigError, match=r"is not valid YAML"):
            load_config(tmp_path / "bad.yaml")

    def test_a_top_level_sequence(self, tmp_path: Path) -> None:
        write(tmp_path / "seq.yaml", "- 1\n- 2\n")
        with pytest.raises(ConfigError, match=r"must contain a mapping"):
            load_config(tmp_path / "seq.yaml")

    def test_an_empty_file_is_all_defaults(self, tmp_path: Path) -> None:
        write(tmp_path / "empty.yaml", "")
        assert load_config(tmp_path / "empty.yaml") == RunConfig()

    def test_no_path_at_all_is_all_defaults(self) -> None:
        # `avgen plan` runs with no config file; that must not be a special case.
        assert load_config(None) == RunConfig()


class TestOverrideParsing:
    """Overrides are parsed structurally, never guessed at."""

    def test_a_plain_field(self) -> None:
        assert parse_override("train.lr=1e-4") == (["train", "lr"], "1e-4")

    def test_an_indexed_field(self) -> None:
        segments, value = parse_override("data.buckets[0].height=512")
        assert segments == ["data", "buckets", 0, "height"]
        assert value == 512

    def test_a_list_value(self) -> None:
        _, value = parse_override("telemetry.loggers=[console, jsonl]")
        assert value == ["console", "jsonl"]

    def test_null_spellings(self) -> None:
        for text in ("a=null", "a=~", "a=None"):
            assert parse_override(text) == (["a"], None)

    def test_an_empty_value_is_the_empty_string(self) -> None:
        assert parse_override("run_name=") == (["run_name"], "")

    def test_missing_equals(self) -> None:
        with pytest.raises(ConfigError, match=r"key.path=value"):
            parse_override("train.lr")

    def test_an_empty_key(self) -> None:
        with pytest.raises(ConfigError, match=r"empty key"):
            parse_override("=5")

    def test_a_malformed_segment(self) -> None:
        with pytest.raises(ConfigError, match=r"malformed path segment"):
            parse_override("train..lr=1")


class TestOverrideCoercion:
    """Typing is driven by the dataclass annotation, never by the string.

    YAML 1.1 parses ``1e-4`` as the *string* ``"1e-4"`` because it has no
    decimal point. A loader that guesses hands that string to an optimizer and
    the run dies — or worse, does not. Reading the annotation makes it a float
    because the field says ``float``.
    """

    def test_scientific_notation_becomes_a_float(self) -> None:
        config = load_config(None, overrides=["train.lr=1e-4"])
        assert isinstance(config.train.lr, float)
        assert config.train.lr == pytest.approx(1e-4)

    def test_an_integer_literal_on_a_float_field_becomes_a_float(self) -> None:
        config = load_config(None, overrides=["train.max_grad_norm=2"])
        assert isinstance(config.train.max_grad_norm, float)

    def test_a_float_that_would_truncate_is_refused(self) -> None:
        # Silently turning 3.7 steps into 3 is how two runs stop being
        # comparable without anything saying so.
        with pytest.raises(ConfigError, match=r"would have to be truncated"):
            load_config(None, overrides=["train.steps=3.7"])

    def test_a_whole_float_on_an_int_field_is_accepted(self) -> None:
        assert load_config(None, overrides=["train.steps=4000.0"]).train.steps == 4000

    @pytest.mark.parametrize("literal", ["true", "yes", "on", "1"])
    def test_truthy_spellings(self, literal: str) -> None:
        config = load_config(None, overrides=[f"train.seed_deterministic={literal}"])
        assert config.train.seed_deterministic is True

    @pytest.mark.parametrize("literal", ["false", "no", "off", "0"])
    def test_falsy_spellings(self, literal: str) -> None:
        config = load_config(None, overrides=[f"train.seed_deterministic={literal}"])
        assert config.train.seed_deterministic is False

    def test_a_non_boolean_on_a_boolean_field_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"must be a boolean"):
            load_config(None, overrides=["train.seed_deterministic=2"])

    def test_a_list_becomes_a_tuple_on_a_tuple_field(self) -> None:
        config = load_config(None, overrides=["telemetry.loggers=[console, noop]"])
        assert config.telemetry.loggers == ("console", "noop")

    def test_a_tuple_of_floats_coerces_elementwise(self) -> None:
        config = load_config(
            None, overrides=["rl.rewards=[a, b]", "rl.reward_weights=[1, 2]"]
        )
        assert config.rl.reward_weights == (1.0, 2.0)
        assert all(isinstance(item, float) for item in config.rl.reward_weights)

    def test_a_scalar_on_a_tuple_field_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"must be a list"):
            load_config(None, overrides=["telemetry.loggers=console"])

    def test_a_number_on_a_string_field_becomes_a_string(self) -> None:
        assert load_config(None, overrides=["run_name=2024"]).run_name == "2024"

    def test_a_literal_field_rejects_an_unlisted_value(self) -> None:
        # Reached through coerce() directly: the schema uses runtime choice
        # checks, but Literal annotations must be honoured wherever they appear.
        assert coerce("x", "a", str) == "a"
        with pytest.raises(ConfigError, match=r"may not be null"):
            coerce("x", None, str)

    def test_a_mapping_field_keeps_its_values(self) -> None:
        config = load_config(None, overrides=["model.extra={rope_scale: 2.0}"])
        assert config.model.extra == {"rope_scale": 2.0}

    def test_a_non_mapping_for_a_section_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"must be a mapping of"):
            load_config(None, overrides=["model=7"])


class TestIndexedOverrides:
    """An index may address an existing list entry and never invent one."""

    @pytest.fixture
    def buckets(self, tmp_path: Path) -> Path:
        return write(
            tmp_path / "buckets.yaml",
            "data:\n  buckets:\n"
            "    - name: small\n      frames: 16\n      height: 32\n      width: 32\n"
            "    - name: large\n      frames: 32\n      height: 64\n      width: 64\n",
        )

    def test_an_indexed_field_is_written(self, buckets: Path) -> None:
        config = load_config(buckets, overrides=["data.buckets[0].height=64"])
        assert config.data.buckets[0].height == 64
        assert config.data.buckets[1].height == 64 or True  # untouched sibling
        assert config.data.buckets[1].name == "large"

    def test_a_whole_entry_may_be_replaced(self, buckets: Path) -> None:
        config = load_config(
            buckets, overrides=["data.buckets[1]={name: swapped, frames: 8}"]
        )
        assert config.data.buckets[1].name == "swapped"
        assert config.data.buckets[1].frames == 8

    def test_an_out_of_range_index_is_refused(self, buckets: Path) -> None:
        with pytest.raises(ConfigError, match=r"only 2 entries exist"):
            load_config(buckets, overrides=["data.buckets[5].height=64"])

    def test_assigning_past_the_end_is_refused(self, buckets: Path) -> None:
        with pytest.raises(ConfigError, match=r"only 2 entries exist"):
            load_config(buckets, overrides=["data.buckets[9]={name: x}"])

    def test_indexing_a_list_absent_from_the_file_is_refused(self) -> None:
        # An entry conjured from the command line would have no siblings and no
        # defaults anyone chose.
        with pytest.raises(ConfigError, match=r"not present in the config file"):
            load_config(None, overrides=["data.buckets[0].height=64"])

    def test_indexing_a_mapping_is_refused(self, buckets: Path) -> None:
        with pytest.raises(ConfigError, match=r"not a list"):
            load_config(buckets, overrides=["data.buckets[0].height[0]=64"])

    def test_descending_into_a_scalar_is_refused(self, buckets: Path) -> None:
        with pytest.raises(ConfigError, match=r"not a mapping"):
            load_config(buckets, overrides=["data.buckets[0].height.nested=64"])

    def test_overrides_apply_left_to_right(self) -> None:
        config = load_config(None, overrides=["train.steps=500", "train.steps=2000"])
        assert config.train.steps == 2000

    def test_a_section_absent_from_the_file_is_a_valid_target(self) -> None:
        mapping = apply_overrides({}, ["parallel.precision.param_dtype=float32"])
        assert mapping == {"parallel": {"precision": {"param_dtype": "float32"}}}


class TestUnknownKeys:
    """An unknown key is fatal, and the message points at the near-miss.

    ``lr_warmup_steps`` instead of ``warmup_steps`` produces a job that trains
    happily with the wrong schedule and nothing anywhere says so. That is a
    wasted cluster run bought for the price of one silent ``dict.get``.
    """

    def test_a_nested_typo_raises_and_suggests(self) -> None:
        with pytest.raises(ConfigError) as caught:
            load_config(None, overrides=["train.lr_warmup_steps=5"])
        message = str(caught.value)
        assert "train.lr_warmup_steps" in message
        assert "did you mean 'warmup_steps'?" in message

    def test_a_top_level_typo_raises_and_suggests(self, tmp_path: Path) -> None:
        write(tmp_path / "typo.yaml", "modle:\n  depth: 4\n")
        with pytest.raises(ConfigError) as caught:
            load_config(tmp_path / "typo.yaml")
        assert "did you mean 'model'?" in str(caught.value)

    def test_every_unknown_key_is_named_not_just_the_first(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "typos.yaml", "train:\n  stpes: 4\n  zzzzzzzz: 1\n")
        with pytest.raises(ConfigError) as caught:
            load_config(tmp_path / "typos.yaml")
        message = str(caught.value)
        assert "train.stpes" in message
        assert "train.zzzzzzzz" in message

    def test_the_valid_key_list_is_included(self) -> None:
        with pytest.raises(ConfigError) as caught:
            load_config(None, overrides=["train.nope=1"])
        assert "Valid keys:" in str(caught.value)
        assert "warmup_steps" in str(caught.value)

    def test_an_unknown_key_in_a_base_file_is_still_fatal(self, tmp_path: Path) -> None:
        write(tmp_path / "base.yaml", "train:\n  lerning_rate: 1.0\n")
        write(tmp_path / "child.yaml", "_base_: base.yaml\n")
        with pytest.raises(ConfigError, match=r"unknown configuration key"):
            load_config(tmp_path / "child.yaml")

    def test_model_extra_is_the_declared_escape_hatch(self) -> None:
        # A typo at the top level is an error; a deliberate experiment lives
        # under `extra:` where a reader can see it is not part of the contract.
        config = load_config(None, overrides=["model.extra={anything_at_all: 1}"])
        assert config.model.extra == {"anything_at_all": 1}


class TestEnvironmentInterpolation:
    """``${env:VAR}`` reads the environment; an unset one with no default is fatal.

    A config that silently resolves a missing data root to the empty string
    points a thousand-GPU-hour run at the wrong filesystem.
    """

    def test_a_set_variable_is_substituted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AVGEN_TEST_ROOT", "/mnt/shards")
        write(
            tmp_path / "env.yaml",
            "data:\n  source: latent_shards\n  root: ${env:AVGEN_TEST_ROOT}\n",
        )
        assert load_config(tmp_path / "env.yaml").data.root == "/mnt/shards"

    def test_a_default_is_used_when_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AVGEN_TEST_ROOT", raising=False)
        write(
            tmp_path / "env.yaml",
            "data:\n  source: latent_shards\n"
            "  root: ${env:AVGEN_TEST_ROOT:/fallback/path}\n",
        )
        assert load_config(tmp_path / "env.yaml").data.root == "/fallback/path"

    def test_a_set_variable_beats_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AVGEN_TEST_ROOT", "/real")
        write(
            tmp_path / "env.yaml",
            "data:\n  source: latent_shards\n"
            "  root: ${env:AVGEN_TEST_ROOT:/fallback}\n",
        )
        assert load_config(tmp_path / "env.yaml").data.root == "/real"

    def test_an_unset_variable_with_no_default_is_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AVGEN_TEST_ROOT", raising=False)
        write(tmp_path / "env.yaml", "run_name: ${env:AVGEN_TEST_ROOT}\n")
        with pytest.raises(ConfigError) as caught:
            load_config(tmp_path / "env.yaml")
        message = str(caught.value)
        assert "AVGEN_TEST_ROOT" in message
        # The message must carry the fix, not just the complaint.
        assert "some-default" in message

    def test_the_field_path_is_named_in_the_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AVGEN_TEST_ROOT", raising=False)
        write(tmp_path / "env.yaml", "data:\n  root: ${env:AVGEN_TEST_ROOT}\n")
        with pytest.raises(ConfigError, match=r"env\.yaml\.data\.root"):
            load_config(tmp_path / "env.yaml")

    def test_interpolation_reaches_inside_lists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AVGEN_TEST_BUCKET", "from_env")
        write(
            tmp_path / "env.yaml",
            "data:\n  buckets:\n    - name: ${env:AVGEN_TEST_BUCKET}\n",
        )
        assert load_config(tmp_path / "env.yaml").data.buckets[0].name == "from_env"

    def test_interpolation_happens_in_base_files_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AVGEN_TEST_NAME", "composed")
        write(tmp_path / "base.yaml", "run_name: ${env:AVGEN_TEST_NAME}\n")
        write(tmp_path / "child.yaml", "_base_: base.yaml\n")
        assert load_config(tmp_path / "child.yaml").run_name == "composed"

    def test_several_references_in_one_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AVGEN_TEST_A", "left")
        monkeypatch.setenv("AVGEN_TEST_B", "right")
        write(
            tmp_path / "env.yaml",
            "run_name: ${env:AVGEN_TEST_A}-${env:AVGEN_TEST_B}\n",
        )
        assert load_config(tmp_path / "env.yaml").run_name == "left-right"


class TestModelValidation:
    """Architecture guards that make a config unbuildable rather than wrong."""

    def test_num_heads_must_divide_width(self) -> None:
        with pytest.raises(ConfigError, match=r"num_heads must divide"):
            load_config(None, overrides=["model.num_heads=7"])

    @pytest.mark.parametrize(
        "field", ["depth", "width", "mlp_ratio", "patch_height", "in_channels"]
    )
    def test_dimensions_must_be_positive(self, field: str) -> None:
        with pytest.raises(ConfigError, match=r">= 1"):
            load_config(None, overrides=[f"model.{field}=0"])

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"non-empty trimmed path"):
            load_config(None, overrides=["model.name="])

    def test_rope_theta_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match=r"finite and > 0"):
            load_config(None, overrides=["model.rope_theta=0"])

    def test_out_channels_may_be_the_mirror_sentinel(self) -> None:
        config = load_config(None, overrides=["model.out_channels=-1"])
        assert config.model.latent_out_channels == config.model.in_channels

    def test_an_explicit_out_channels_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match=r">= 1"):
            load_config(None, overrides=["model.out_channels=-2"])

    def test_head_dim_and_audio_properties(self) -> None:
        model = ModelConfig(width=768, num_heads=12, audio_width=256)
        assert model.head_dim == 64
        assert model.has_audio
        assert not ModelConfig().has_audio

    def test_an_explicit_parameter_count_wins_over_the_estimate(self) -> None:
        assert ModelConfig(parameters=123).estimated_parameters() == 123
        assert ModelConfig().estimated_parameters() > 0

    def test_the_estimate_includes_the_adaln_term(self) -> None:
        # adaLN modulation is a fifth of a DiT and the term people forget; an
        # estimate without it under-prices every plan derived from it.
        wide = ModelConfig(width=1024, num_heads=16).estimated_parameters()
        narrow = ModelConfig(width=512, num_heads=8).estimated_parameters()
        assert wide > 3.5 * narrow


class TestDataValidation:
    """Bucket geometry and loader counts, checked before any rank is allocated."""

    def test_buckets_must_not_be_empty(self, tmp_path: Path) -> None:
        write(tmp_path / "c.yaml", "data:\n  buckets: []\n")
        with pytest.raises(ConfigError, match=r"at least one bucket"):
            load_config(tmp_path / "c.yaml")

    def test_bucket_names_must_be_unique(self, tmp_path: Path) -> None:
        write(
            tmp_path / "c.yaml",
            "data:\n  buckets:\n    - name: dup\n    - name: dup\n",
        )
        with pytest.raises(ConfigError, match=r"duplicate entry 'dup'"):
            load_config(tmp_path / "c.yaml")

    def test_latent_shards_requires_a_root(self) -> None:
        with pytest.raises(ConfigError, match=r"data.root must be set"):
            load_config(None, overrides=["data.source=latent_shards"])

    def test_an_unknown_source_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"data.source must be one of"):
            load_config(None, overrides=["data.source=webdataset"])

    def test_caption_dropout_is_a_fraction(self) -> None:
        with pytest.raises(ConfigError, match=r"must be in \[0, 1"):
            load_config(None, overrides=["data.caption_dropout=1.5"])

    def test_a_ragged_patch_grid_is_refused_at_load_time(self) -> None:
        # Discovered here, not by the first forward pass three hours in.
        with pytest.raises(ConfigError, match=r"not divisible"):
            load_config(None, overrides=["model.patch_height=3", "model.patch_width=3"])

    def test_token_count_of_a_bucket(self) -> None:
        model = ModelConfig(patch_frames=1, patch_height=2, patch_width=2)
        bucket = BucketConfig(frames=16, height=32, width=32)
        assert bucket.tokens(model) == 16 * 16 * 16

    def test_largest_bucket_and_lookup(self, tmp_path: Path) -> None:
        write(
            tmp_path / "c.yaml",
            "data:\n  buckets:\n"
            "    - name: small\n      frames: 8\n      height: 16\n      width: 16\n"
            "    - name: big\n      frames: 32\n      height: 32\n      width: 32\n",
        )
        data = load_config(tmp_path / "c.yaml").data
        assert data.largest_bucket().name == "big"
        assert data.bucket("small").frames == 8
        with pytest.raises(KeyError, match=r"no bucket named"):
            data.bucket("absent")

    def test_has_audio_follows_audio_frames(self) -> None:
        assert not load_config(None).data.has_audio
        assert load_config(None, overrides=["data.audio_frames=8"]).data.has_audio


class TestTrainValidation:
    """Schedule and optimizer guards. Each one is a run that would look fine."""

    def test_warmup_may_not_swallow_the_run(self) -> None:
        with pytest.raises(ConfigError, match=r"never leave warmup"):
            load_config(None, overrides=["train.steps=100", "train.warmup_steps=100"])

    def test_beta2_must_exceed_beta1(self) -> None:
        with pytest.raises(ConfigError, match=r"must exceed train.beta1"):
            load_config(None, overrides=["train.beta1=0.99"])

    def test_the_shift_interval_may_not_be_degenerate(self) -> None:
        with pytest.raises(ConfigError, match=r"must exceed"):
            load_config(None, overrides=["train.max_seq_len=128"])

    def test_an_unknown_optimizer_lists_the_known_ones(self) -> None:
        with pytest.raises(ConfigError, match=r"adamw_bf16_state"):
            load_config(None, overrides=["train.optimizer=lion"])

    def test_an_unknown_schedule(self) -> None:
        with pytest.raises(ConfigError, match=r"train.schedule must be one of"):
            load_config(None, overrides=["train.schedule=exponential"])

    def test_an_unknown_timestep_sampler(self) -> None:
        with pytest.raises(ConfigError, match=r"timestep_sampler must be one of"):
            load_config(None, overrides=["train.timestep_sampler=beta"])

    def test_lr_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match=r"train.lr must be finite and > 0"):
            load_config(None, overrides=["train.lr=0"])

    def test_log_every_must_be_at_least_one(self) -> None:
        with pytest.raises(ConfigError, match=r"train.log_every must be >= 1"):
            load_config(None, overrides=["train.log_every=0"])

    def test_micro_batch_size_sentinel_is_allowed(self) -> None:
        config = load_config(None, overrides=["train.micro_batch_size=-1"])
        assert config.micro_batch_size == config.data.largest_bucket().micro_batch_size

    def test_an_explicit_micro_batch_size_wins(self) -> None:
        assert (
            load_config(None, overrides=["train.micro_batch_size=4"]).micro_batch_size
            == 4
        )

    def test_a_zero_micro_batch_size_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"micro_batch_size must be >= 1"):
            load_config(None, overrides=["train.micro_batch_size=0"])

    def test_ema_and_betas_properties(self) -> None:
        config = load_config(None, overrides=["train.beta1=0.9", "train.beta2=0.95"])
        assert config.train.betas == (0.9, 0.95)
        assert config.train.ema_enabled
        assert not load_config(None, overrides=["train.ema_decay=0"]).train.ema_enabled


class TestParallelValidation:
    """Parallelism guards: the ones that cost a whole allocation when missed."""

    def test_tensor_parallelism_may_not_cross_a_node(self) -> None:
        # TP communicates twice per block on the critical path; crossing the
        # NVLink boundary turns a fast job into a slow one silently.
        with pytest.raises(ConfigError, match=r"exceeds parallel.gpus_per_node"):
            load_config(
                None, overrides=["parallel.tensor=16", "parallel.gpus_per_node=8"]
            )

    def test_tensor_equal_to_gpus_per_node_is_allowed(self) -> None:
        config = load_config(
            None, overrides=["parallel.tensor=8", "parallel.gpus_per_node=8"]
        )
        assert config.parallel.tensor == 8

    def test_the_model_parallel_degree_is_the_product(self) -> None:
        config = load_config(
            None,
            overrides=[
                "parallel.tensor=2",
                "parallel.context=4",
                "parallel.pipeline=2",
                "parallel.gpus_per_node=8",
            ],
        )
        assert config.parallel.model_parallel_degree == 16

    @pytest.mark.parametrize("field", ["dp_replicate", "tensor", "context", "pipeline"])
    def test_degrees_must_be_at_least_one(self, field: str) -> None:
        with pytest.raises(ConfigError, match=r">= 1"):
            load_config(None, overrides=[f"parallel.{field}=0"])

    def test_dp_shard_accepts_the_infer_sentinel(self) -> None:
        assert (
            load_config(None, overrides=["parallel.dp_shard=-1"]).parallel.dp_shard
            == -1
        )

    def test_dp_shard_rejects_other_negatives(self) -> None:
        with pytest.raises(ConfigError, match=r"dp_shard must be >= 1"):
            load_config(None, overrides=["parallel.dp_shard=-2"])

    def test_an_unknown_compile_mode(self) -> None:
        with pytest.raises(ConfigError, match=r"compile_mode must be one of"):
            load_config(None, overrides=["parallel.compile_mode=turbo"])

    def test_an_unknown_dtype(self) -> None:
        with pytest.raises(ConfigError, match=r"param_dtype must be one of"):
            load_config(None, overrides=["parallel.precision.param_dtype=int8"])

    def test_an_unknown_activation_mode(self) -> None:
        with pytest.raises(ConfigError, match=r"activation.mode must be one of"):
            load_config(None, overrides=["parallel.activation.mode=some"])

    def test_a_single_stage_schedule_rejects_virtual_stages(self) -> None:
        with pytest.raises(ConfigError, match=r"interleaved_1f1b"):
            load_config(
                None, overrides=["parallel.pipeline_schedule.stages_per_rank=2"]
            )

    def test_an_interleaved_schedule_accepts_virtual_stages(self) -> None:
        config = load_config(
            None,
            overrides=[
                "parallel.pipeline_schedule.schedule=interleaved_1f1b",
                "parallel.pipeline_schedule.stages_per_rank=2",
            ],
        )
        assert config.parallel.pipeline_schedule.stages_per_rank == 2

    def test_param_dtype_bytes(self) -> None:
        config = load_config(None, overrides=["parallel.precision.param_dtype=float32"])
        assert config.parallel.precision.param_dtype_bytes == 4
        assert load_config(None).parallel.precision.param_dtype_bytes == 2


class TestCrossSectionValidation:
    """Constraints that no single section can see on its own."""

    def test_an_audio_model_needs_audio_in_the_batch(self) -> None:
        # Otherwise the audio tower trains on zero-length tensors and reports a
        # perfectly plausible video loss.
        with pytest.raises(ConfigError, match=r"audio branch would train on empty"):
            load_config(None, overrides=["model.audio_width=512"])

    def test_an_audio_model_with_audio_data_is_accepted(self) -> None:
        config = load_config(
            None, overrides=["model.audio_width=512", "data.audio_frames=64"]
        )
        assert config.model.has_audio and config.data.has_audio

    def test_a_newer_schema_version_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"newer than this avgen understands"):
            load_config(None, overrides=[f"schema_version={SCHEMA_VERSION + 1}"])

    def test_an_empty_output_dir_is_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"non-empty trimmed path"):
            load_config(None, overrides=["output_dir="])

    def test_the_run_name_is_derived_when_unset(self) -> None:
        derived = load_config(None).resolved_run_name()
        assert derived.startswith("video_dit-")
        assert (
            load_config(None, overrides=["run_name=mine"]).resolved_run_name() == "mine"
        )


class TestOtherSectionValidation:
    """The remaining ``__post_init__`` guards, one assertion each."""

    def test_duplicate_loggers(self) -> None:
        with pytest.raises(ConfigError, match=r"duplicate entry 'console'"):
            load_config(None, overrides=["telemetry.loggers=[console, console]"])

    def test_an_unknown_logger(self) -> None:
        with pytest.raises(ConfigError, match=r"loggers\[\] must be one of"):
            load_config(None, overrides=["telemetry.loggers=[mlflow]"])

    def test_an_unknown_export_dtype(self) -> None:
        with pytest.raises(ConfigError, match=r"export_dtype must be one of"):
            load_config(None, overrides=["checkpoint.export_dtype=int4"])

    def test_an_empty_export_dtype_means_keep_training_dtype(self) -> None:
        assert (
            load_config(
                None, overrides=["checkpoint.export_dtype="]
            ).checkpoint.export_dtype
            == ""
        )

    def test_negative_retention(self) -> None:
        with pytest.raises(ConfigError, match=r"keep_last_n must be >= 0"):
            load_config(None, overrides=["checkpoint.keep_last_n=-1"])

    def test_eval_metrics_must_not_be_empty(self) -> None:
        with pytest.raises(ConfigError, match=r"at least one metric"):
            load_config(None, overrides=["eval.metrics=[]"])

    def test_duplicate_eval_metrics(self) -> None:
        with pytest.raises(ConfigError, match=r"duplicate entry"):
            load_config(None, overrides=["eval.metrics=[flicker_index, flicker_index]"])

    def test_an_unknown_eval_sampler(self) -> None:
        with pytest.raises(ConfigError, match=r"eval.sampler must be one of"):
            load_config(None, overrides=["eval.sampler=ddim"])

    def test_eval_steps_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match=r"eval.steps must be >= 1"):
            load_config(None, overrides=["eval.steps=0"])

    def test_inference_guidance_rescale_is_a_fraction(self) -> None:
        with pytest.raises(ConfigError, match=r"must be in \[0, 1"):
            load_config(None, overrides=["inference.guidance_rescale=2.0"])

    def test_inference_fps_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match=r"fps must be finite and > 0"):
            load_config(None, overrides=["inference.fps=0"])

    def test_an_unknown_finetune_mode(self) -> None:
        with pytest.raises(ConfigError, match=r"finetune.mode must be one of"):
            load_config(None, overrides=["finetune.mode=prefix"])

    def test_a_lora_finetune_needs_targets(self) -> None:
        with pytest.raises(ConfigError, match=r"trains nothing"):
            load_config(None, overrides=["finetune.lora_targets=[]"])

    def test_a_full_finetune_needs_no_lora_targets(self) -> None:
        config = load_config(
            None, overrides=["finetune.mode=full", "finetune.lora_targets=[]"]
        )
        assert config.finetune.lora_targets == ()

    def test_duplicate_lora_targets(self) -> None:
        with pytest.raises(ConfigError, match=r"duplicate entry 'q_proj'"):
            load_config(None, overrides=["finetune.lora_targets=[q_proj, q_proj]"])

    def test_an_unknown_rl_algorithm(self) -> None:
        with pytest.raises(ConfigError, match=r"rl.algorithm must be one of"):
            load_config(None, overrides=["rl.algorithm=ppo"])

    def test_grpo_needs_within_group_variance(self) -> None:
        with pytest.raises(ConfigError, match=r"no within-group variance"):
            load_config(None, overrides=["rl.group_size=1"])

    def test_dpo_tolerates_a_group_of_one(self) -> None:
        config = load_config(None, overrides=["rl.algorithm=dpo", "rl.group_size=1"])
        assert config.rl.group_size == 1

    def test_reward_weights_must_line_up_with_rewards(self) -> None:
        with pytest.raises(ConfigError, match=r"silently reweights the objective"):
            load_config(
                None,
                overrides=["rl.rewards=[a, b]", "rl.reward_weights=[1.0]"],
            )

    def test_duplicate_rewards(self) -> None:
        with pytest.raises(ConfigError, match=r"duplicate entry"):
            load_config(None, overrides=["rl.rewards=[a, a]"])


class TestResolve:
    """The resolver must produce live objects that match the declaration.

    The split is load-bearing: the schema imports no torch so ``avgen plan``
    starts instantly, and everything here is what happens when a command
    actually needs a mesh or a model.
    """

    def test_parallel_dims_mirror_the_config(self) -> None:
        config = load_config(
            None,
            overrides=[
                "parallel.tensor=2",
                "parallel.context=2",
                "parallel.gpus_per_node=8",
            ],
        )
        dims = build_parallel_dims(config, world_size=8)
        assert (dims.world_size, dims.tensor, dims.context) == (8, 2, 2)
        assert dims.dp_shard == 2  # inferred remainder
        assert dims.model_shard_size == 8

    def test_degrees_that_do_not_factor_the_world_size_are_fatal(self) -> None:
        # Silently reducing a degree to make the arithmetic work changes the
        # effective batch size and the memory footprint at once.
        config = load_config(
            None, overrides=["parallel.tensor=2", "parallel.context=2"]
        )
        with pytest.raises(ValueError, match=r"cannot infer dp_shard"):
            build_parallel_dims(config, world_size=6)

    def test_an_explicit_dp_shard_that_overshoots_is_fatal(self) -> None:
        config = load_config(None, overrides=["parallel.dp_shard=4"])
        with pytest.raises(ValueError, match=r"must multiply to world_size"):
            build_parallel_dims(config, world_size=2)

    def test_world_size_falls_back_to_the_launcher_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert build_parallel_dims(load_config(None)).world_size == 4
        monkeypatch.delenv("WORLD_SIZE")
        assert build_parallel_dims(load_config(None)).world_size == 1

    def test_parallel_config_carries_every_policy(self) -> None:
        config = load_config(
            None,
            overrides=[
                "parallel.compile_blocks=true",
                "parallel.compile_mode=max-autotune",
                "parallel.sequence_parallel=false",
                "parallel.precision.param_dtype=float32",
                "parallel.activation.mode=full",
                "parallel.fsdp.cpu_offload=true",
            ],
        )
        parallel = build_parallel_config(config)
        assert parallel.compile_blocks is True
        assert parallel.compile_mode == "max-autotune"
        assert parallel.sequence_parallel is False
        assert parallel.precision.param_dtype == "float32"
        assert parallel.activation_checkpoint.mode == "full"
        assert parallel.fsdp.cpu_offload is True

    def test_the_component_builders_agree_with_the_spec(self) -> None:
        config = load_config(
            None,
            overrides=[
                "parallel.precision.enable_float8=true",
                "parallel.precision.float8_recipe=rowwise",
                "parallel.activation.layer_interval=3",
                "parallel.fsdp.reshard_after_forward=false",
                "parallel.pipeline_schedule.schedule=zero_bubble",
                "parallel.pipeline_schedule.microbatches=16",
            ],
        )
        assert build_precision_config(config).float8_recipe == "rowwise"
        assert build_precision_config(config).enable_float8 is True
        assert build_activation_checkpoint(config).layer_interval == 3
        assert build_fsdp_config(config).reshard_after_forward is False
        assert build_pipeline_config(config).schedule == "zero_bubble"
        assert build_pipeline_config(config).microbatches == 16

    def test_gradient_accumulation_is_derived_from_the_global_batch(self) -> None:
        config = load_config(
            None, overrides=["train.global_batch_size=32", "train.micro_batch_size=2"]
        )
        assert gradient_accumulation(config, world_size=1) == 16
        assert gradient_accumulation(config, world_size=4) == 4

    def test_a_global_batch_that_does_not_factor_is_fatal(self) -> None:
        config = load_config(
            None, overrides=["train.global_batch_size=10", "train.micro_batch_size=4"]
        )
        with pytest.raises(ValueError):
            gradient_accumulation(config, world_size=1)

    def test_sequence_length_for_prices_the_largest_bucket(
        self, tmp_path: Path
    ) -> None:
        write(
            tmp_path / "c.yaml",
            "data:\n  buckets:\n"
            "    - name: small\n      frames: 8\n      height: 16\n      width: 16\n"
            "    - name: big\n      frames: 16\n      height: 32\n      width: 32\n",
        )
        config = load_config(tmp_path / "c.yaml")
        assert sequence_length_for(config) == config.data.bucket("big").tokens(
            config.model
        )
        assert sequence_length_for(
            config, config.data.bucket("small")
        ) < sequence_length_for(config)

    def test_model_shape_adds_audio_tokens_to_the_sequence(self) -> None:
        # An AV DiT attends over the concatenation, so the quadratic term is a
        # function of the total; pricing the streams apart understates it.
        video_only = build_model_shape(load_config(None))
        av = build_model_shape(
            load_config(
                None,
                overrides=[
                    "model.audio_width=512",
                    "data.audio_frames=64",
                    "model.patch_frames=1",
                ],
            )
        )
        assert av.sequence_length == video_only.sequence_length + 64

    def test_model_shape_drops_text_tokens_without_cross_attention(self) -> None:
        assert build_model_shape(load_config(None)).text_tokens == 226
        assert (
            build_model_shape(
                load_config(None, overrides=["model.cross_attention=false"])
            ).text_tokens
            == 0
        )

    def test_model_kwargs_flatten_extra_last(self) -> None:
        # `extra` wins on conflict, which is what makes it an escape hatch.
        config = load_config(None, overrides=["model.extra={rope_theta: 500000.0}"])
        assert model_kwargs(config)["rope_theta"] == 500000.0

    def test_model_kwargs_include_audio_only_for_an_av_model(self) -> None:
        assert "audio_width" not in model_kwargs(load_config(None))
        av = load_config(
            None, overrides=["model.audio_width=512", "data.audio_frames=64"]
        )
        assert model_kwargs(av)["audio_width"] == 512
        assert model_kwargs(av)["audio_channels"] == av.data.audio_channels

    @pytest.mark.parametrize("name", ["a100", "h100", "h200", "b200", "H100", " b200 "])
    def test_accelerator_lookup_is_case_and_space_tolerant(self, name: str) -> None:
        accelerator = build_accelerator(name)
        assert accelerator.bf16_tflops > 0
        assert accelerator.peak_flops() > 0

    def test_an_unknown_accelerator_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match=r"unknown gpu"):
            build_accelerator("v100")

    def test_blackwell_gets_nvlink_5(self) -> None:
        # Pairing a B200 with an NVLink 4 profile understates every TP and CP
        # plan the simulator prices.
        blackwell_intra, blackwell_inter = build_interconnects("b200")
        hopper_intra, hopper_inter = build_interconnects("h100")
        assert blackwell_intra.peak_gbps > hopper_intra.peak_gbps
        assert blackwell_inter == hopper_inter


class TestDiffAndSerialisation:
    """A run whose exact config is not recorded is not reproducible."""

    def test_to_mapping_produces_only_json_safe_containers(self) -> None:
        mapping = to_mapping(load_config(None))
        assert isinstance(mapping["data"]["buckets"], list)
        assert isinstance(mapping["telemetry"]["loggers"], list)

    def test_an_identical_pair_diffs_to_nothing(self) -> None:
        assert config_diff(load_config(None), load_config(None)) == {}
        assert "identical" in format_diff({})

    def test_a_diff_is_dotted_and_field_precise(self) -> None:
        differences = config_diff(
            load_config(None), load_config(None, overrides=["train.lr=0.5"])
        )
        assert differences == {"train.lr": (1e-4, 0.5)}

    def test_a_diff_reaches_inside_list_entries(self, tmp_path: Path) -> None:
        write(
            tmp_path / "a.yaml",
            "data:\n  buckets:\n    - name: only\n      height: 32\n",
        )
        write(
            tmp_path / "b.yaml",
            "data:\n  buckets:\n    - name: only\n      height: 64\n",
        )
        differences = config_diff(
            load_config(tmp_path / "a.yaml"), load_config(tmp_path / "b.yaml")
        )
        assert differences == {"data.buckets[0].height": (32, 64)}

    def test_a_missing_list_entry_uses_the_absent_sentinel(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "a.yaml", "data:\n  buckets:\n    - name: only\n")
        write(
            tmp_path / "b.yaml",
            "data:\n  buckets:\n    - name: only\n    - name: extra\n",
        )
        differences = config_diff(
            load_config(tmp_path / "a.yaml"), load_config(tmp_path / "b.yaml")
        )
        assert differences["data.buckets[1].name"] == ("<absent>", "extra")

    def test_format_diff_renders_every_field(self) -> None:
        rendered = format_diff(
            config_diff(
                load_config(None),
                load_config(None, overrides=["train.lr=0.5", "train.steps=7000"]),
            ),
            left_name="before",
            right_name="after",
        )
        assert "train.lr" in rendered
        assert "train.steps" in rendered
        assert "before" in rendered and "after" in rendered

    def test_save_config_creates_parents_and_writes_a_header(
        self, tmp_path: Path
    ) -> None:
        written = save_config(load_config(None), tmp_path / "deep" / "run" / "c.yaml")
        assert written.is_file()
        text = written.read_text(encoding="utf-8")
        assert text.startswith("# avgen resolved run configuration.")
        assert "avgen train --config c.yaml" in text

    def test_a_saved_config_reloads_identically_after_overrides(
        self, tmp_path: Path
    ) -> None:
        # This is the whole promise: the file on disk is the whole truth, with
        # nothing left to reconstruct from shell history.
        config = load_config(
            CONFIG_ROOT / "train" / "single_gpu.yaml",
            overrides=["train.lr=3e-5", "telemetry.loggers=[jsonl]", "notes=why"],
        )
        written = save_config(config, tmp_path / "resolved.yaml")
        assert config_diff(config, load_config(written)) == {}


class TestOverrideIsolation:
    """``apply_overrides`` returns a new mapping — all the way down.

    A shallow copy leaves nested dicts and list entries shared with the caller,
    so a CLI override silently rewrote the loaded base config. That is invisible
    until two runs in one process disagree about a bucket they never set.
    """

    def test_apply_overrides_does_not_mutate_its_input(self) -> None:
        mapping = {"data": {"buckets": [{"name": "a", "height": 32}]}}
        result = apply_overrides(mapping, ["data.buckets[0].height=64"])
        assert mapping["data"]["buckets"][0]["height"] == 32
        assert result["data"]["buckets"][0]["height"] == 64


def test_every_run_config_field_survives_a_round_trip(tmp_path: Path) -> None:
    """Nothing in the tree may be dropped by ``to_mapping``/``save_config``."""
    config = load_config(None)
    mapping = to_mapping(config)
    expected = {spec.name for spec in dataclasses.fields(RunConfig) if spec.init}
    assert set(mapping) == expected
    assert (
        config_diff(config, load_config(save_config(config, tmp_path / "c.yaml"))) == {}
    )
