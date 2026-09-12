"""Tests for the ``avgen`` command line.

The CLI is the only part of avgen a person types, so its contract is behavioural
rather than structural: which exit code a shell script can branch on, whether an
error is one readable line or forty frames of Python, and whether the three
commands that must work on a broken install (``--help``, ``info``, ``plan``)
still work.

Every test drives :func:`avgen.cli.main.main` directly with an argv list rather
than spawning a subprocess. Same code path, no process startup, and a failing
assertion points at a line instead of at a return code.

CPU only: the ``force_cpu`` fixture hides any visible GPU so a machine with one
runs the same test as a machine without.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml

from avgen.cli._common import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_USAGE,
    human_bytes,
    human_count,
    human_seconds,
    rule,
    table,
)
from avgen.cli.main import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SMOKE = str(CONFIGS / "train" / "smoke_cpu.yaml")

PLAN_ARGV = [
    "plan",
    "--world-size",
    "512",
    "--seq-len",
    "65536",
    "--params",
    "2e9",
    "--depth",
    "32",
    "--width",
    "2560",
]

COMMANDS = (
    "train",
    "finetune",
    "rl",
    "generate",
    "eval",
    "simulate",
    "plan",
    "checkpoint",
    "data",
    "info",
)


@dataclass(frozen=True)
class Invocation:
    """What one ``avgen`` invocation returned and printed."""

    code: int
    out: str
    err: str


@pytest.fixture
def force_cpu(monkeypatch):
    """Hide any visible GPU so the CLI resolves to CPU everywhere."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def cli(capsys):
    """Return a callable that runs ``main(argv)`` and captures its output."""

    def run(*argv: str) -> Invocation:
        try:
            code = main(list(argv))
        except SystemExit as stop:  # argparse exits rather than returning
            code = 0 if stop.code is None else int(stop.code)
        captured = capsys.readouterr()
        return Invocation(code=code, out=captured.out, err=captured.err)

    return run


def losses_from(output: str) -> list[float]:
    """Pull the ``loss`` column out of the trainer's console table."""
    lines = [line for line in output.splitlines() if line.strip()]
    header = next(line for line in lines if line.split()[:1] == ["step"])
    column = header.split().index("loss")
    start = lines.index(header) + 2  # skip the rule under the header
    values: list[float] = []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) <= column or not fields[0].isdigit():
            continue
        values.append(float(fields[column]))
    return values


class TestEntryPoints:
    def test_a_checkpoint_failure_exits_cleanly_rather_than_crashing(
        self, cli, monkeypatch, tmp_path
    ) -> None:
        """DCP's CheckpointException derives from BaseException, not Exception.

        That makes it invisible to every ordinary handler, including main's, so
        a save or load failure — the failure a long training job is most likely
        to hit — would leave the process with a raw traceback and whatever exit
        code Python picked. Verified through the real class, not a stand-in.
        """
        from torch.distributed.checkpoint.api import CheckpointException

        from avgen.cli import checkpoint as checkpoint_command

        assert not issubclass(CheckpointException, Exception)

        def explode(_arguments):
            raise CheckpointException("write", {0: (RuntimeError("disk full"), None)})

        monkeypatch.setattr(checkpoint_command, "run", explode)
        result = cli("checkpoint", "inspect", str(tmp_path))
        assert result.code == EXIT_ERROR
        assert "avgen" in result.err

    def test_a_genuine_base_exception_is_not_swallowed(
        self, cli, monkeypatch, tmp_path
    ) -> None:
        """The BaseException catch is one exemption, not a blanket."""
        from avgen.cli import checkpoint as checkpoint_command

        def explode(_arguments):
            raise SystemExit(7)

        monkeypatch.setattr(checkpoint_command, "run", explode)
        # The cli fixture converts SystemExit to its code; main must not have
        # turned it into an error line.
        assert cli("checkpoint", "inspect", str(tmp_path)).code == 7

    def test_help_exits_zero_and_names_the_starting_commands(self, cli) -> None:
        result = cli("--help")
        assert result.code == EXIT_OK
        assert "usage: avgen" in result.out
        for command in COMMANDS:
            assert command in result.out

    @pytest.mark.parametrize("command", COMMANDS)
    def test_every_subcommand_documents_itself(self, cli, command) -> None:
        result = cli(command, "--help")
        assert result.code == EXIT_OK
        assert f"usage: avgen {command}" in result.out

    def test_version_prints_and_stops(self, cli) -> None:
        result = cli("--version")
        assert result.code == EXIT_OK
        assert result.out.startswith("avgen ")

    def test_no_command_prints_help_and_reports_a_usage_error(self, cli) -> None:
        # Help on stdout so it can be piped; exit 2 so a script can tell that
        # nothing ran.
        result = cli()
        assert result.code == EXIT_USAGE
        assert "usage: avgen" in result.out

    def test_every_subcommand_registers_a_handler(self) -> None:
        # A subcommand without a handler falls through to the help path and
        # exits 2, which looks like a user error and is not one.
        parser = build_parser()
        for action in parser._subparsers._group_actions:
            for name, subparser in action.choices.items():
                assert subparser.get_default("handler") is not None, name

    def test_info_reports_the_environment_and_never_fails(self, cli) -> None:
        # A diagnostic that fails is not a diagnostic: problems appear in the
        # output, not in the exit code.
        result = cli("info")
        assert result.code == EXIT_OK
        for section in ("VERSIONS", "OPTIONAL EXTRAS", "SUBSYSTEMS", "NEXT"):
            assert section in result.out
        assert "torch" in result.out
        assert "avgen.data" in result.out

    def test_info_can_list_the_evaluation_metrics(self, cli) -> None:
        result = cli("info", "--metrics")
        assert result.code == EXIT_OK
        assert "EVALUATION METRICS" in result.out


class TestPlan:
    def test_plan_needs_no_config_and_no_gpu(self, cli, monkeypatch) -> None:
        # This is the command you run before booking the cluster. It must work
        # with no config file, no dataset, and no CUDA.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = cli(*PLAN_ARGV)
        assert result.code == EXIT_OK
        assert "plans fit in memory" in result.out
        assert "RANKED BY PREDICTED THROUGHPUT" in result.out
        assert "2.00 B parameters" in result.out
        assert "512 x H100 SXM" in result.out

    def test_plan_emits_machine_readable_candidates(self, cli) -> None:
        result = cli(*PLAN_ARGV, "--json", "--top-k", "3")
        assert result.code == EXIT_OK
        payload = json.loads(result.out)
        assert payload["world_size"] == 512
        assert payload["shape"]["parameters"] == 2_000_000_000
        assert payload["fitting_plans"] >= len(payload["candidates"])
        assert 1 <= len(payload["candidates"]) <= 3

    def test_plan_reports_when_nothing_fits(self, cli) -> None:
        # Exit 1, not 0 with an empty table: a CI job gating on a config change
        # has to be able to tell.
        result = cli(
            "plan",
            "--world-size",
            "8",
            "--seq-len",
            "131072",
            "--params",
            "5e11",
            "--depth",
            "64",
            "--width",
            "8192",
        )
        assert result.code == EXIT_ERROR
        assert "0 plans fit in memory" in result.out

    def test_plan_rejects_an_impossible_shape(self, cli) -> None:
        assert cli("plan", "--world-size", "0").code == EXIT_ERROR
        assert cli("plan", "--world-size", "8", "--depth", "0").code == EXIT_ERROR
        bad_heads = cli("plan", "--world-size", "8", "--width", "100", "--heads", "7")
        assert bad_heads.code == EXIT_ERROR
        assert "does not divide" in bad_heads.err


class TestExitCodes:
    def test_the_documented_codes_are_distinct(self) -> None:
        codes = (EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_CONFIG, EXIT_INTERRUPT)
        assert codes == (0, 1, 2, 3, 130)
        assert len(set(codes)) == len(codes)

    def test_an_unknown_subcommand_is_a_usage_error(self, cli) -> None:
        result = cli("trian")
        assert result.code == EXIT_USAGE
        assert "invalid choice" in result.err

    def test_a_missing_config_file_is_a_configuration_error(self, cli) -> None:
        result = cli("train", "--config", "/no/such/file.yaml", "--dry-run")
        assert result.code == EXIT_CONFIG
        assert "config file not found" in result.err

    def test_an_unknown_override_key_is_a_configuration_error(self, cli) -> None:
        # Never a silent no-op: a typo'd override is a wasted cluster run.
        result = cli("train", "--config", SMOKE, "train.lr_rate=1e-4", "--dry-run")
        assert result.code == EXIT_CONFIG
        assert "unknown configuration key" in result.err

    def test_a_malformed_override_is_a_configuration_error(self, cli) -> None:
        result = cli("train", "--config", SMOKE, "train.lr", "--dry-run")
        assert result.code == EXIT_CONFIG

    def test_a_missing_required_argument_is_a_usage_error(self, cli) -> None:
        assert cli("train", "--dry-run").code == EXIT_USAGE
        assert cli("plan").code == EXIT_USAGE

    def test_a_subcommand_group_without_an_action_is_a_runtime_error(self, cli) -> None:
        for group in ("data", "checkpoint"):
            result = cli(group)
            assert result.code == EXIT_ERROR
            assert "needs an action" in result.err

    def test_a_path_that_does_not_exist_is_a_runtime_error(self, cli, tmp_path) -> None:
        missing = str(tmp_path / "nope")
        assert cli("data", "inspect", missing).code == EXIT_ERROR
        assert cli("checkpoint", "inspect", missing).code == EXIT_ERROR


class TestErrorRendering:
    def test_an_error_prints_one_line_and_no_traceback(self, cli, monkeypatch) -> None:
        # A stack trace is for a bug in avgen. Printing forty frames for a user
        # error teaches people to stop reading error messages.
        def explode(_arguments):
            raise RuntimeError("the cluster is on fire")

        monkeypatch.setattr("avgen.cli.info.run", explode)
        result = cli("info")
        assert result.code == EXIT_ERROR
        assert "Traceback" not in result.err
        assert len(result.err.strip().splitlines()) == 1
        assert result.err.startswith("avgen: error: the cluster is on fire")
        assert "--traceback" in result.err

    def test_an_error_with_no_message_still_names_its_type(
        self, cli, monkeypatch
    ) -> None:
        def explode(_arguments):
            raise RuntimeError

        monkeypatch.setattr("avgen.cli.info.run", explode)
        result = cli("info")
        assert result.code == EXIT_ERROR
        assert "RuntimeError" in result.err

    def test_the_traceback_flag_restores_the_frames(self, cli, monkeypatch) -> None:
        def explode(_arguments):
            raise RuntimeError("the cluster is on fire")

        monkeypatch.setattr("avgen.cli.info.run", explode)
        with pytest.raises(RuntimeError, match="on fire"):
            cli("--traceback", "info")

    def test_a_keyboard_interrupt_is_not_a_crash(self, cli, monkeypatch) -> None:
        # A training run stopped on purpose is not a crash, and a traceback for
        # one trains people to ignore tracebacks.
        def interrupt(_arguments):
            raise KeyboardInterrupt

        monkeypatch.setattr("avgen.cli.info.run", interrupt)
        result = cli("info")
        assert result.code == EXIT_INTERRUPT
        assert result.err.strip() == "avgen: interrupted"
        assert "Traceback" not in result.err

    def test_a_config_error_raised_by_a_handler_keeps_its_own_code(
        self, cli, monkeypatch
    ) -> None:
        from avgen.config.loader import ConfigError

        def explode(_arguments):
            raise ConfigError("bucket 3 is wider than it is tall")

        monkeypatch.setattr("avgen.cli.info.run", explode)
        result = cli("info")
        assert result.code == EXIT_CONFIG
        assert "wider than it is tall" in result.err


class TestTrain:
    def test_dry_run_validates_and_stops_before_allocating(
        self, cli, tmp_path, force_cpu
    ) -> None:
        output = tmp_path / "run"
        result = cli(
            "train", "--config", SMOKE, "--dry-run", "--output-dir", str(output)
        )
        assert result.code == EXIT_OK
        assert "configuration is valid; stopping before allocation" in result.out
        assert "smoke-cpu" in result.out
        # Nothing at all was written: --dry-run stops before the resolved config.
        assert not output.exists()

    @pytest.mark.parametrize(
        ("command", "config"),
        (
            ("train", "train/smoke_cpu.yaml"),
            ("finetune", "finetune/lora.yaml"),
            ("rl", "rl/dpo.yaml"),
        ),
    )
    def test_every_training_entry_point_has_a_dry_run(
        self, cli, tmp_path, command, config
    ) -> None:
        result = cli(
            command,
            "--config",
            str(CONFIGS / config),
            "--dry-run",
            "--output-dir",
            str(tmp_path / command),
        )
        assert result.code == EXIT_OK
        assert f"avgen {command}" in result.out
        assert "stopping before allocation" in result.out

    def test_print_config_emits_loadable_yaml(self, cli, tmp_path) -> None:
        result = cli("train", "--config", SMOKE, "--print-config")
        assert result.code == EXIT_OK
        mapping = yaml.safe_load(result.out)
        assert mapping["run_name"] == "smoke-cpu"
        assert mapping["train"]["steps"] == 20
        assert mapping["model"]["depth"] == 4

        # The real test of "resolved": feed it back in with no bases to compose.
        resolved = tmp_path / "resolved.yaml"
        resolved.write_text(yaml.safe_dump(mapping))
        again = cli("train", "--config", str(resolved), "--print-config")
        assert again.code == EXIT_OK
        assert yaml.safe_load(again.out) == mapping

    def test_a_command_line_override_takes_effect(self, cli) -> None:
        result = cli(
            "train",
            "--config",
            SMOKE,
            "train.lr=3e-4",
            "train.steps=7",
            "--dry-run",
        )
        assert result.code == EXIT_OK
        assert "lr 0.0003" in result.out
        assert "7 steps" in result.out

    def test_train_runs_end_to_end_and_writes_the_resolved_config(
        self, cli, tmp_path, force_cpu
    ) -> None:
        output = tmp_path / "run"
        result = cli(
            "train",
            "--config",
            SMOKE,
            "--output-dir",
            str(output),
            "train.steps=3",
            "train.log_every=1",
        )
        assert result.code == EXIT_OK, result.err
        # The resolved config is what makes a run reproducible six months later.
        written = output / "config.yaml"
        assert written.is_file()
        assert yaml.safe_load(written.read_text())["train"]["steps"] == 3
        assert "resolved config written to" in result.out

        values = losses_from(result.out)
        assert len(values) == 3, result.out
        assert all(math.isfinite(value) and value > 0.0 for value in values)

    def test_finetune_refuses_to_start_without_a_base(
        self, cli, tmp_path, force_cpu
    ) -> None:
        # A fine-tune without a base is just training, and the two need
        # different learning rates, schedules, and EMA settings.
        result = cli(
            "finetune",
            "--config",
            str(CONFIGS / "finetune" / "lora.yaml"),
            "--output-dir",
            str(tmp_path / "ft"),
        )
        assert result.code == EXIT_ERROR
        assert "finetune.base_checkpoint is empty" in result.err

    def test_resume_overrides_the_config(self, cli, tmp_path) -> None:
        result = cli(
            "train",
            "--config",
            SMOKE,
            "--resume",
            str(tmp_path / "somewhere"),
            "--print-config",
        )
        assert result.code == EXIT_OK
        mapping = yaml.safe_load(result.out)
        assert mapping["checkpoint"]["resume"] == str(tmp_path / "somewhere")


class TestDataCommands:
    def synthesize(self, cli, tmp_path, samples: int = 4):
        destination = tmp_path / "shards"
        result = cli(
            "data",
            "synthesize",
            "--config",
            SMOKE,
            "--output",
            str(destination),
            "--samples",
            str(samples),
        )
        assert result.code == EXIT_OK, result.err
        return destination, result

    def test_synthesize_inspect_validate_round_trip(self, cli, tmp_path) -> None:
        destination, written = self.synthesize(cli, tmp_path)
        assert f"wrote 4 samples to {destination}" in written.out
        # A shard is a directory, and the marker is written last.
        shard = destination / "shard-00000"
        for name in ("data.safetensors", "manifest.json", "COMMIT"):
            assert (shard / name).is_file(), name

        inspected = cli("data", "inspect", str(destination))
        assert inspected.code == EXIT_OK
        assert "1 shard(s)" in inspected.out
        assert "shard-00000" in inspected.out
        assert "synthetic-video-v1" in inspected.out

        validated = cli("data", "validate", str(destination))
        assert validated.code == EXIT_OK
        assert "all checks passed" in validated.out
        assert "samples checked  4" in validated.out

    def test_validate_catches_a_corrupted_container(self, cli, tmp_path) -> None:
        destination, _ = self.synthesize(cli, tmp_path)
        container = destination / "shard-00000" / "data.safetensors"
        payload = bytearray(container.read_bytes())
        payload[-1] ^= 0x01
        container.write_bytes(bytes(payload))

        result = cli("data", "validate", str(destination))
        assert result.code == EXIT_ERROR
        assert "failures         1" in result.out
        assert "Do not train on this" in result.out

    def test_an_uncommitted_shard_is_invisible(self, cli, tmp_path) -> None:
        # Deliberately invisible: a half-written shard must never be read.
        destination, _ = self.synthesize(cli, tmp_path)
        (destination / "shard-00000" / "COMMIT").unlink()

        inspected = cli("data", "inspect", str(destination))
        assert inspected.code == EXIT_ERROR
        assert "no committed shards" in inspected.err
        assert cli("data", "validate", str(destination)).code == EXIT_ERROR

    def test_validate_checks_shapes_against_the_configured_buckets(
        self, cli, tmp_path
    ) -> None:
        destination, _ = self.synthesize(cli, tmp_path)
        matching = cli("data", "validate", str(destination), "--config", SMOKE)
        assert matching.code == EXIT_OK

        wrong = tmp_path / "wrong-buckets.yaml"
        wrong.write_text(
            yaml.safe_dump(
                {
                    "data": {
                        "buckets": [
                            {
                                "name": "mismatched",
                                "frames": 8,
                                "height": 64,
                                "width": 64,
                                "micro_batch_size": 1,
                            }
                        ]
                    }
                }
            )
        )
        mismatched = cli("data", "validate", str(destination), "--config", str(wrong))
        assert mismatched.code == EXIT_ERROR
        assert "matches no configured bucket" in mismatched.out

    def test_the_shard_action_explains_why_it_will_not_guess(
        self, cli, tmp_path
    ) -> None:
        result = cli("data", "shard", str(tmp_path), str(tmp_path / "out"))
        assert result.code == EXIT_ERROR
        assert "ALREADY ENCODED" in result.err
        assert "avgen data synthesize" in result.err

    def test_a_synthesized_corpus_is_trainable(self, cli, tmp_path, force_cpu) -> None:
        # The whole point of synthesize: every other command is runnable end to
        # end with no dataset, no decoder, and no GPU.
        destination, _ = self.synthesize(cli, tmp_path, samples=8)
        result = cli(
            "train",
            "--config",
            SMOKE,
            "--output-dir",
            str(tmp_path / "run"),
            "data.source=latent_shards",
            f"data.root={destination}",
            "train.steps=3",
            "train.log_every=1",
        )
        assert result.code == EXIT_OK, result.err
        assert all(math.isfinite(value) for value in losses_from(result.out))


class TestSimulate:
    def test_simulate_prices_the_configured_run(self, cli) -> None:
        result = cli("simulate", "--config", SMOKE)
        assert result.code == EXIT_OK
        assert "avgen simulation" in result.out
        assert "MEMORY" in result.out
        assert "THROUGHPUT" in result.out
        # dp_shard: -1 does not pin a world size, and defaulting silently would
        # report a simulation for a scale nobody asked about.
        assert "parallel.dp_shard is -1" in result.out

    def test_simulate_builds_a_real_mesh_under_a_fake_process_group(self, cli) -> None:
        result = cli(
            "simulate",
            "--config",
            SMOKE,
            "--world-size",
            "4",
            "--topology",
            "--rank",
            "2",
            "parallel.dp_shard=4",
            "train.global_batch_size=8",
        )
        assert result.code == EXIT_OK, result.err
        assert "real DeviceMesh over a FakeProcessGroup" in result.out
        assert "4 ranks, impersonating rank 2" in result.out
        assert "dp_shard=4" in result.out
        assert "byte-identical batches" in result.out

    def test_simulate_writes_a_machine_readable_report(self, cli, tmp_path) -> None:
        destination = tmp_path / "report.json"
        result = cli(
            "simulate", "--config", SMOKE, "--json", "--save", str(destination)
        )
        assert result.code == EXIT_OK
        payload = json.loads(result.out)
        assert payload["run_name"] == "smoke-cpu"
        assert payload["model_name"] == "video_dit"
        assert payload["bucket"] == "tiny"
        assert payload["fits"] is True
        assert destination.is_file()
        assert (
            json.loads(destination.read_text())["world_size"] == payload["world_size"]
        )

    def test_simulate_refuses_a_world_size_the_config_cannot_factor(self, cli) -> None:
        result = cli(
            "simulate", "--config", SMOKE, "--world-size", "4", "parallel.dp_shard=4"
        )
        assert result.code == EXIT_ERROR
        assert "global_batch_size" in result.err


class TestCheckpointCommand:
    def make_checkpoint(self, tmp_path):
        from torch import nn

        from avgen.checkpoint import CheckpointManager
        from avgen.core import RNGStreams, TrainState
        from avgen.train import build_optimizer, build_schedule

        torch.manual_seed(0)
        model = nn.Linear(8, 8)
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
        state = TrainState(
            model=model,
            optimizer=optimizer,
            rng=RNGStreams.from_seed(0),
            schedule=build_schedule(
                "constant", optimizer, total_steps=8, warmup_steps=1
            ),
        )
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        directory = manager.save(3, state, async_save=False)
        manager.close()
        return directory

    def test_inspect_describes_a_checkpoint_without_loading_it(
        self, cli, tmp_path
    ) -> None:
        directory = self.make_checkpoint(tmp_path)
        result = cli("checkpoint", "inspect", str(directory))
        assert result.code == EXIT_OK
        assert "is_dcp        True" in result.out
        assert "avgen_checkpoint.json" in result.out
        assert "no EMA weights detected" in result.out

    def test_inspect_emits_machine_readable_output(self, cli, tmp_path) -> None:
        directory = self.make_checkpoint(tmp_path)
        result = cli("checkpoint", "inspect", str(directory), "--json")
        assert result.code == EXIT_OK
        payload = json.loads(result.out)
        assert payload["exists"] is True
        assert payload["is_dcp"] is True
        assert payload["size_bytes"] > 0
        assert ".metadata" in payload["entries"]

    def test_convert_rewrites_the_storage_format(self, cli, tmp_path) -> None:
        directory = self.make_checkpoint(tmp_path)
        destination = tmp_path / "release"
        result = cli(
            "checkpoint",
            "convert",
            str(directory),
            str(destination),
            "--world-size",
            "8",
        )
        assert result.code == EXIT_OK, result.err
        assert (destination / "model.safetensors").is_file()
        assert "changes the storage format, not the shard count" in result.out

    def make_model_checkpoint(self, tmp_path, *, ema: bool = False):
        """Save a checkpoint from a real registered model, config beside it.

        Export builds the model before writing it, so unlike the other verbs it
        needs an architecture — which means a checkpoint of an ``nn.Linear``
        cannot exercise it.
        """
        from avgen.checkpoint import CheckpointManager
        from avgen.config import load_config, save_config
        from avgen.config.resolve import model_kwargs
        from avgen.core import RNGStreams, TrainState
        from avgen.models import build_model
        from avgen.train import ShardedEMA, build_optimizer, build_schedule

        config = load_config(
            None,
            overrides=[
                "model.name=video_dit",
                "model.depth=1",
                "model.width=32",
                "model.num_heads=2",
            ],
        )
        torch.manual_seed(0)
        model = build_model(config.model.name, model_kwargs(config))
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
        state = TrainState(
            model=model,
            optimizer=optimizer,
            rng=RNGStreams.from_seed(0),
            schedule=build_schedule(
                "constant", optimizer, total_steps=8, warmup_steps=1
            ),
            ema=ShardedEMA(model, decay=0.99) if ema else None,
        )
        manager = CheckpointManager(tmp_path / "ckm", keep_last_n=0)
        directory = manager.save(3, state, async_save=False)
        manager.close()
        save_config(config, directory / "config.yaml")
        return directory

    def test_export_writes_an_inference_artifact(self, cli, tmp_path) -> None:
        directory = self.make_model_checkpoint(tmp_path)
        destination = tmp_path / "weights"
        result = cli("checkpoint", "export", str(directory), str(destination))
        assert result.code == EXIT_OK, result.err
        assert (destination / "model.safetensors").is_file()
        assert "cannot be resumed from" in result.out

    def test_export_says_which_weights_it_took(self, cli, tmp_path) -> None:
        # Silently exporting the live weights when the caller wanted the average
        # ships a visibly worse model with nothing in the output to show it.
        directory = self.make_model_checkpoint(tmp_path, ema=True)
        result = cli(
            "checkpoint", "export", str(directory), str(tmp_path / "ema"), "--ema"
        )
        assert result.code == EXIT_OK, result.err
        assert "loaded EMA weights" in result.out

        plain = cli("checkpoint", "export", str(directory), str(tmp_path / "raw"))
        assert plain.code == EXIT_OK, plain.err
        assert "live training weights" in plain.out

    def test_export_refuses_a_checkpoint_with_no_ema(self, cli, tmp_path) -> None:
        directory = self.make_model_checkpoint(tmp_path, ema=False)
        result = cli(
            "checkpoint", "export", str(directory), str(tmp_path / "ema"), "--ema"
        )
        assert result.code != EXIT_OK

    def test_export_without_an_architecture_says_so(self, cli, tmp_path) -> None:
        # The failure to avoid is loading weights into a differently shaped
        # model and reporting a shape error instead of the missing config.
        directory = self.make_checkpoint(tmp_path)  # no config.yaml beside it
        result = cli("checkpoint", "export", str(directory), str(tmp_path / "out"))
        assert result.code != EXIT_OK
        assert "config.yaml" in result.out + result.err


class TestGenerateAndEval:
    def test_generate_can_print_its_pinned_settings_without_a_model(
        self, cli, tmp_path
    ) -> None:
        # Two clips generated at different steps or guidance are not a
        # comparison of anything, so the settings are the output that matters.
        result = cli(
            "generate",
            "--checkpoint",
            str(tmp_path / "ckpt"),
            "--prompt",
            "a cat knocking a glass off a table",
            "--steps",
            "12",
            "--print-settings",
        )
        assert result.code == EXIT_OK, result.err
        assert "avgen generate" in result.out
        assert "prompts       1" in result.out
        assert "x12" in result.out

    def test_generate_requires_exactly_one_prompt_source(self, cli, tmp_path) -> None:
        assert cli("generate", "--checkpoint", str(tmp_path)).code == EXIT_USAGE
        missing = cli(
            "generate",
            "--checkpoint",
            str(tmp_path),
            "--prompt-file",
            str(tmp_path / "nope.txt"),
        )
        assert missing.code == EXIT_ERROR
        assert "prompt file not found" in missing.err

    def test_eval_lists_its_metrics(self, cli) -> None:
        result = cli("eval", "--list-metrics")
        assert result.code == EXIT_OK
        assert "DEPENDENCY-FREE METRICS" in result.out

    def test_eval_says_what_it_needs(self, cli, tmp_path) -> None:
        empty = cli("eval")
        assert empty.code == EXIT_ERROR
        assert "--checkpoint is required" in empty.err

        orphan = cli("eval", "--report", str(tmp_path / "report.json"))
        assert orphan.code == EXIT_ERROR
        assert "needs --baseline" in orphan.err


class TestFormatting:
    def test_counts_get_a_readable_suffix(self) -> None:
        assert human_count(2.05e9) == "2.05 B"
        assert human_count(1.5e12) == "1.50 T"
        assert human_count(999) == "999"

    def test_bytes_are_binary_units(self) -> None:
        assert human_bytes(80 * 1024**3) == "80.0 GiB"
        assert human_bytes(512) == "512 B"

    def test_durations_keep_three_significant_figures(self) -> None:
        assert human_seconds(1.8342) == "1.83 s"
        assert human_seconds(0.4123) == "412.3 ms"
        assert human_seconds(1.2e-5) == "12 us"

    def test_a_table_is_sized_to_its_content(self) -> None:
        # A plan table with a truncated bottleneck column hides the single most
        # useful field in the output.
        rendered = table(
            (("plan", "plan"), ("bottleneck", "bottleneck")),
            [
                {"plan": "dp8", "bottleneck": "fsdp.all_gather"},
                {"plan": "dp4tp2", "bottleneck": "attention"},
            ],
        )
        lines = rendered.splitlines()
        assert "fsdp.all_gather" in rendered
        assert len(lines) == 4
        assert set(lines[1]) <= {"-", " "}
        assert table((("a", "a"),), []) == "(no rows)"

    def test_a_rule_can_carry_a_title(self) -> None:
        assert rule(width=10) == "-" * 10
        titled = rule("PLAN", width=20)
        assert titled.startswith("-- PLAN ")
        assert len(titled) == 20
