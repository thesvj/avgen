"""Tests for checkpoint retention, resume, and release-artifact export.

Three properties, each of which fails silently everywhere else:

* **retention never deletes the newest complete checkpoint.** A retention bug
  turns a preemption from a lost hour into a lost run, and the misconfiguration
  that makes it possible (``keep_last_n=0`` meaning "keep none") is easy to
  write.
* **an incomplete checkpoint is invisible.** A directory without a marker is the
  debris of a preempted writer; resuming from one restores half a model.
* **an exported artifact round-trips.** Export is a one-way door — the training
  state is gone — so a lossy export is discovered when the weights are needed,
  not when they are written.

CPU-only, single process, tiny shapes.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from avgen.checkpoint import (
    CHECKPOINT_DIR_PREFIX,
    CHECKPOINT_FORMAT_VERSION,
    INDEX_FILENAME,
    MARKER_FILENAME,
    WEIGHTS_FILENAME,
    CheckpointManager,
    convert,
    export_safetensors,
    import_safetensors,
)
from avgen.checkpoint.stateful import MODEL_OPTIMIZER_KEY
from avgen.core import RNGStreams, TrainState
from avgen.train import build_optimizer, build_schedule


class TinyModel(nn.Module):
    """Two small linear layers — enough state to shard, cheap enough to spam."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.head = nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


def make_state(seed: int = 0) -> TrainState:
    torch.manual_seed(seed)
    model = TinyModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    return TrainState(
        model=model,
        optimizer=optimizer,
        rng=RNGStreams.from_seed(seed),
        schedule=build_schedule("constant", optimizer, total_steps=16, warmup_steps=1),
    )


def take_a_step(state: TrainState) -> None:
    """Put real values in the optimizer moments so a resume has something to do."""
    loss = state.model(torch.randn(2, 8)).square().mean()
    loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)


def save_steps(manager: CheckpointManager, state: TrainState, steps) -> None:
    for step in steps:
        state.step = step
        manager.save(step, state, async_save=False)
    manager.wait()


def step_directory(root, step: int):
    return root / f"{CHECKPOINT_DIR_PREFIX}{step:010d}"


class TestRetention:
    def test_keep_last_n_and_keep_every_n_steps_compose(self, tmp_path) -> None:
        # The usual production setting: a short rolling window for preemption
        # recovery plus a sparse permanent series for post-hoc analysis. The
        # sparse series must survive the rolling window's pruning.
        manager = CheckpointManager(
            tmp_path / "ck", keep_last_n=2, keep_every_n_steps=5
        )
        save_steps(manager, make_state(), [1, 2, 3, 4, 5, 6, 7, 10])
        assert [entry.step for entry in manager.list_checkpoints()] == [5, 7, 10]
        assert manager.latest_step() == 10

    def test_keep_last_n_zero_means_unbounded_not_delete_everything(
        self, tmp_path
    ) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [1, 2, 3])
        assert [entry.step for entry in manager.list_checkpoints()] == [1, 2, 3]

    def test_the_newest_complete_checkpoint_is_never_deleted(self, tmp_path) -> None:
        # The one mistake a retention policy must never make.
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=1)
        save_steps(manager, make_state(), [1, 2, 3, 4])
        entries = manager.list_checkpoints()
        assert [entry.step for entry in entries] == [4]
        assert entries[-1].path.is_dir()
        assert (entries[-1].path / MARKER_FILENAME).is_file()

    def test_abandoned_partial_directories_are_swept_up(self, tmp_path) -> None:
        # One per preemption, full-size, never resumable. On a large model that
        # is terabytes a month.
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [1])
        orphan = step_directory(manager.root, 0)
        orphan.mkdir()
        (orphan / "__0_0.distcp").write_bytes(b"garbage")
        save_steps(manager, make_state(), [2])
        assert not orphan.exists()

    def test_a_partial_directory_newer_than_the_last_complete_one_is_left_alone(
        self, tmp_path
    ) -> None:
        # It may be an upload still in flight, in this process or a concurrent
        # evaluation job.
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [1, 2])
        inflight = step_directory(manager.root, 9)
        inflight.mkdir()
        save_steps(manager, make_state(), [3])
        assert inflight.exists()


class TestDiscovery:
    def test_an_unmarked_directory_is_invisible(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [4])
        incomplete = step_directory(manager.root, 99)
        incomplete.mkdir()
        (incomplete / "__0_0.distcp").write_bytes(b"half a checkpoint")
        assert [entry.step for entry in manager.list_checkpoints()] == [4]
        assert manager.latest_step() == 4

    def test_an_unparseable_marker_is_not_a_marker(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [4])
        (step_directory(manager.root, 4) / MARKER_FILENAME).write_text("{not json")
        assert manager.list_checkpoints() == ()
        assert manager.latest_step() is None

    def test_a_marker_that_is_not_a_mapping_is_rejected(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [4])
        (step_directory(manager.root, 4) / MARKER_FILENAME).write_text("[1, 2]")
        assert manager.latest_step() is None

    def test_unrelated_directories_are_ignored(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [4])
        (manager.root / "step_notanumber").mkdir()
        (manager.root / "tensorboard").mkdir()
        (manager.root / "notes.txt").write_text("hello")
        assert [entry.step for entry in manager.list_checkpoints()] == [4]

    def test_scanning_a_directory_that_does_not_exist_is_not_an_error(
        self, tmp_path
    ) -> None:
        manager = CheckpointManager(tmp_path / "never-written")
        assert manager.list_checkpoints() == ()
        assert manager.latest_step() is None


class TestSaveAndLoad:
    def test_a_fresh_run_resumes_from_nothing(self, tmp_path) -> None:
        # None is the normal answer for a fresh run, so a trainer can call load
        # unconditionally.
        assert CheckpointManager(tmp_path / "ck").load(make_state()) is None

    def test_resume_restores_weights_optimizer_and_progress(self, tmp_path) -> None:
        state = make_state()
        take_a_step(state)
        state.step = 6
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        manager.save(6, state, async_save=False)
        manager.wait()

        expected = {k: v.clone() for k, v in state.model.state_dict().items()}
        key = next(iter(state.optimizer.state))
        moment = state.optimizer.state[key]["exp_avg"].clone()

        fresh = make_state(seed=11)
        take_a_step(fresh)
        assert manager.load(fresh) == 6
        for name, value in fresh.model.state_dict().items():
            assert torch.equal(value, expected[name]), name
        fresh_key = next(iter(fresh.optimizer.state))
        assert torch.allclose(fresh.optimizer.state[fresh_key]["exp_avg"], moment)
        assert fresh.step == 6

    def test_an_explicit_step_that_does_not_exist_is_an_error(self, tmp_path) -> None:
        # An explicit request that cannot be satisfied is an error; an implicit
        # one is a fresh start.
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [2])
        assert manager.load(make_state(), step=2) == 2
        with pytest.raises(FileNotFoundError, match="step 3"):
            manager.load(make_state(), step=3)

    def test_a_newer_format_version_is_refused(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        save_steps(manager, make_state(), [1])
        marker = step_directory(manager.root, 1) / MARKER_FILENAME
        payload = json.loads(marker.read_text())
        payload["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
        marker.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="newer than the supported"):
            manager.load(make_state())

    def test_the_marker_describes_the_checkpoint(self, tmp_path) -> None:
        # This is what makes a checkpoint directory self-describing six months
        # later, when the run that wrote it is gone.
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        state = make_state()
        state.step = 3
        manager.save(3, state, async_save=False, extras={"git_sha": "abc123"})
        manager.wait()
        entry = manager.list_checkpoints()[-1]
        assert entry.step == 3
        assert entry.metadata["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert entry.metadata["extras"] == {"git_sha": "abc123"}
        assert MODEL_OPTIMIZER_KEY in entry.metadata["keys"]
        assert entry.world_size == 1
        assert entry.data_world == 1
        assert "completed_at" in entry.metadata

    def test_a_negative_or_boolean_step_is_refused(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck")
        with pytest.raises(ValueError, match="non-negative integer"):
            manager.save(-1, make_state())
        with pytest.raises(ValueError, match="non-negative integer"):
            manager.save(True, make_state())

    def test_retention_settings_are_validated(self, tmp_path) -> None:
        for kwargs, message in (
            ({"keep_last_n": -1}, "keep_last_n"),
            ({"keep_every_n_steps": -1}, "keep_every_n_steps"),
            ({"save_timeout_s": 0.0}, "save_timeout_s"),
            ({"thread_count": 0}, "thread_count"),
        ):
            with pytest.raises(ValueError, match=message):
                CheckpointManager(tmp_path / "ck", **kwargs)

    def test_the_manager_drains_itself_when_used_as_a_context(self, tmp_path) -> None:
        state = make_state()
        state.step = 1
        with CheckpointManager(tmp_path / "ck", keep_last_n=0) as manager:
            manager.save(1, state, async_save=True)
        assert manager.has_pending_save is False
        assert manager.latest_step() == 1


class TestConcurrentSaves:
    def test_a_second_save_drains_the_first(self, tmp_path) -> None:
        # Exactly one save may be outstanding. Draining first is also what
        # surfaces an exception from the previous save while it can still be
        # acted on, rather than at process exit.
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        state = make_state()
        state.step = 1
        manager.save(1, state, async_save=True)
        assert manager.has_pending_save

        state.step = 2
        manager.save(2, state, async_save=True)
        # The first is complete the moment the second one starts.
        assert (step_directory(manager.root, 1) / MARKER_FILENAME).is_file()
        manager.close()
        assert [entry.step for entry in manager.list_checkpoints()] == [1, 2]
        assert manager.has_pending_save is False

    def test_waiting_with_nothing_pending_is_a_no_op(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck")
        manager.wait()
        manager.wait()
        assert manager.has_pending_save is False

    def test_loading_drains_a_pending_save_first(self, tmp_path) -> None:
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        state = make_state()
        state.step = 5
        manager.save(5, state, async_save=True)
        assert manager.load(make_state(seed=3)) == 5


class TestExport:
    def test_export_import_round_trips_every_tensor(self, tmp_path) -> None:
        state = make_state()
        take_a_step(state)
        directory = export_safetensors(tmp_path / "weights", state.model)
        assert (directory / WEIGHTS_FILENAME).is_file()
        assert (directory / INDEX_FILENAME).is_file()

        restored = TinyModel()
        import_safetensors(directory, restored)
        for name, value in state.model.state_dict().items():
            assert torch.equal(restored.state_dict()[name], value), name

    def test_a_multi_shard_export_indexes_every_file(self, tmp_path) -> None:
        state = make_state()
        # 128 bytes forces one file per parameter; the point is that the index
        # is the thing that makes the shard set readable.
        directory = export_safetensors(
            tmp_path / "weights", state.model, max_shard_bytes=128
        )
        index = json.loads((directory / INDEX_FILENAME).read_text())
        files = set(index["weight_map"].values())
        assert len(files) > 1
        assert set(index["weight_map"]) == set(state.model.state_dict())
        for filename in files:
            assert (directory / filename).is_file()
        assert index["metadata"]["total_size"] > 0

        restored = TinyModel()
        import_safetensors(directory, restored)
        for name, value in state.model.state_dict().items():
            assert torch.equal(restored.state_dict()[name], value), name

    def test_export_casts_floating_point_tensors(self, tmp_path) -> None:
        from safetensors.torch import load_file

        state = make_state()
        directory = export_safetensors(
            tmp_path / "bf16", state.model, dtype=torch.bfloat16
        )
        tensors = load_file(directory / WEIGHTS_FILENAME)
        assert tensors
        assert all(value.dtype is torch.bfloat16 for value in tensors.values())
        for name, value in tensors.items():
            expected = state.model.state_dict()[name].to(torch.bfloat16)
            assert torch.equal(value, expected), name

    def test_export_writes_the_metadata_header(self, tmp_path) -> None:
        from safetensors import safe_open

        directory = export_safetensors(
            tmp_path / "weights", make_state().model, metadata={"run": "smoke"}
        )
        with safe_open(directory / WEIGHTS_FILENAME, framework="pt") as handle:
            header = handle.metadata()
        assert header["run"] == "smoke"
        assert header["format"] == "pt"

    def test_export_rejects_a_non_string_metadata_pair(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="str -> str"):
            export_safetensors(
                tmp_path / "weights", make_state().model, metadata={"steps": 10}
            )

    def test_export_rejects_a_non_positive_shard_size(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="max_shard_bytes"):
            export_safetensors(
                tmp_path / "weights", make_state().model, max_shard_bytes=0
            )

    def test_a_single_file_can_be_imported_directly(self, tmp_path) -> None:
        state = make_state()
        directory = export_safetensors(tmp_path / "weights", state.model)
        restored = TinyModel()
        import_safetensors(directory / WEIGHTS_FILENAME, restored)
        assert torch.equal(restored.encoder.weight, state.model.encoder.weight)

    def test_a_directory_without_an_index_still_loads(self, tmp_path) -> None:
        # A hand-assembled release has no index; refusing it would make the
        # exporter the only way to produce a readable directory.
        state = make_state()
        directory = export_safetensors(tmp_path / "weights", state.model)
        (directory / INDEX_FILENAME).unlink()
        restored = TinyModel()
        import_safetensors(directory, restored)
        assert torch.equal(restored.head.bias, state.model.head.bias)

    def test_an_empty_directory_is_reported_not_guessed_at(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no safetensors shards"):
            import_safetensors(tmp_path / "empty", TinyModel())


class TestConvert:
    def dcp_checkpoint(self, tmp_path):
        state = make_state()
        take_a_step(state)
        state.step = 4
        manager = CheckpointManager(tmp_path / "ck", keep_last_n=0)
        directory = manager.save(4, state, async_save=False)
        manager.close()
        return state, directory

    def test_dcp_converts_to_a_loadable_safetensors_directory(self, tmp_path) -> None:
        state, source = self.dcp_checkpoint(tmp_path)
        destination = convert(source, tmp_path / "release", to="safetensors")
        assert (destination / INDEX_FILENAME).is_file()
        restored = TinyModel()
        import_safetensors(destination, restored)
        for name, value in state.model.state_dict().items():
            assert torch.equal(restored.state_dict()[name], value), name

    def test_conversion_can_cast_on_the_way_out(self, tmp_path) -> None:
        from safetensors.torch import load_file

        _, source = self.dcp_checkpoint(tmp_path)
        destination = convert(
            source, tmp_path / "release", to="safetensors", dtype=torch.bfloat16
        )
        tensors = load_file(destination / WEIGHTS_FILENAME)
        assert all(value.dtype is torch.bfloat16 for value in tensors.values())

    def test_safetensors_converts_back_into_the_training_layout(self, tmp_path) -> None:
        # Wrapped in the training layout so the result is loadable as a warm
        # start; the optimizer half is absent, which a non-strict load tolerates.
        state, source = self.dcp_checkpoint(tmp_path)
        release = convert(source, tmp_path / "release", to="safetensors")
        rebuilt = convert(release, tmp_path / "warm", to="dcp")
        assert (rebuilt / ".metadata").is_file()

        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.format_utils import (
            _EmptyStateDictLoadPlanner,
            _load_state_dict,
        )

        flat: dict = {}
        _load_state_dict(
            flat,
            storage_reader=dcp.FileSystemReader(rebuilt),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        weights = flat[MODEL_OPTIMIZER_KEY]["model"]
        for name, value in state.model.state_dict().items():
            assert torch.equal(weights[name], value), name

    def test_an_unknown_target_format_is_refused(self, tmp_path) -> None:
        _, source = self.dcp_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="must be 'safetensors' or 'dcp'"):
            convert(source, tmp_path / "out", to="onnx")

    def test_a_checkpoint_without_a_model_subtree_is_reported(self, tmp_path) -> None:
        import torch.distributed.checkpoint as dcp

        source = tmp_path / "odd"
        dcp.save(
            {"telemetry": {"loss": torch.zeros(2)}},
            storage_writer=dcp.FileSystemWriter(source, overwrite=True),
            no_dist=True,
        )
        with pytest.raises(KeyError, match="model"):
            convert(source, tmp_path / "out", to="safetensors")
