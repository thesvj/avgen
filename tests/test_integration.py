"""End-to-end tests across subsystem boundaries.

The unit tests check that each subsystem honours its contract. These check that
the contracts actually compose — which is a different question, and the one that
unit tests systematically miss. Every test here crosses at least two ownership
boundaries.

All of it runs on CPU at tiny shapes in a few seconds, so it belongs in the fast
CI subset rather than behind a GPU marker.
"""

from __future__ import annotations

import torch

from avgen.checkpoint import CheckpointManager
from avgen.core import GridPatchifier, RNGStreams, TrainState
from avgen.data import SyntheticConfig, SyntheticSource
from avgen.models import VideoDiT, preset
from avgen.parallel import ParallelDims, parallelize
from avgen.simulate import fake_world, simulate_config
from avgen.train import (
    FlowMatchingConfig,
    FlowMatchingObjective,
    MultiTaskConditioning,
    build_optimizer,
    build_schedule,
    build_timestep_sampler,
    train_step,
)


def build_stack(*, seed: int = 0, lr: float = 3e-3):
    """Construct a complete tiny training stack.

    Returns:
        The synthetic source, the train state, and the objective.
    """
    torch.manual_seed(seed)
    config = preset("tiny")
    source = SyntheticSource(
        SyntheticConfig(
            seed=seed,
            num_samples=32,
            video_channels=config.in_channels,
            frames=8,
            height=8,
            width=8,
            audio_frames=0,
            text_tokens=8,
            text_width=config.text_width,
        )
    )
    model = VideoDiT(config)
    model.init_weights()
    optimizer = build_optimizer(model, lr=lr, weight_decay=0.0)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        rng=RNGStreams.from_seed(seed),
        schedule=build_schedule("wsd", optimizer, total_steps=64, warmup_steps=4),
    )
    objective = FlowMatchingObjective(
        FlowMatchingConfig(),
        build_timestep_sampler("shifted_logit_normal"),
        MultiTaskConditioning(),
    )
    return source, state, objective


def run_steps(source, state, objective, count: int) -> list[float]:
    """Run training steps and return the loss history."""
    patchifier = GridPatchifier()
    losses: list[float] = []
    iterator = iter(source)
    for _ in range(count):
        batch = next(iterator)
        state, metrics = train_step(
            state, batch, objective, patchifier=patchifier, max_grad_norm=1.0
        )
        losses.append(float(metrics.loss))
    return losses


class TestTrainingLoop:
    def test_synthetic_data_is_learnable(self) -> None:
        # If this regresses, either the objective's target is wrong or the
        # synthetic source stopped containing learnable structure. Both are
        # silent failures everywhere else.
        source, state, objective = build_stack()
        losses = run_steps(source, state, objective, 30)
        assert all(loss == loss for loss in losses), "loss went non-finite"
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"loss did not improve: {early:.4f} -> {late:.4f}"

    def test_progress_counters_advance(self) -> None:
        source, state, objective = build_stack()
        run_steps(source, state, objective, 5)
        assert state.step == 5
        assert state.samples_seen > 0
        assert state.tokens_seen > 0

    def test_identical_seeds_reproduce_the_loss_curve(self) -> None:
        # Determinism is a contract, not a nicety: without it no two runs are
        # comparable and no regression is attributable.
        first = run_steps(*build_stack(seed=11), count=8)
        second = run_steps(*build_stack(seed=11), count=8)
        assert first == second

    def test_different_seeds_diverge(self) -> None:
        first = run_steps(*build_stack(seed=1), count=8)
        second = run_steps(*build_stack(seed=2), count=8)
        assert first != second


class TestPatchifyRoundTripThroughModel:
    def test_model_output_matches_the_input_token_geometry(self) -> None:
        source, _state, _ = build_stack()
        batch = next(iter(source))
        patchifier = GridPatchifier()
        stream = patchifier.to_tokens(
            batch.video,
            positions=batch.video_positions,
            mask=batch.video_mask,
            noise_level=torch.rand(batch.spec.batch_size),
        )
        # spec.video_tokens counts *latents*; layout.num_tokens counts *tokens*
        # after 2x2 patching. The ratio is the patch volume, and conflating the
        # two is how a sequence-length budget silently comes out 4x wrong.
        patch_volume = (
            stream.layout.patch_frames
            * stream.layout.patch_height
            * stream.layout.patch_width
        )
        assert stream.layout.num_tokens * patch_volume == batch.spec.video_tokens
        restored = patchifier.to_grid(stream)
        assert restored.shape == batch.video.shape
        assert torch.equal(restored, batch.video)


class TestCheckpointResume:
    def test_resume_restores_weights_optimizer_and_progress(self, tmp_path) -> None:
        source, state, objective = build_stack()
        run_steps(source, state, objective, 6)

        manager = CheckpointManager(tmp_path / "ckpt")
        manager.save(state.step, state)
        manager.wait()

        weights_before = {k: v.clone() for k, v in state.model.state_dict().items()}
        key = next(iter(state.optimizer.state))
        moment_before = state.optimizer.state[key]["exp_avg"].clone()
        step_before = state.step

        # Corrupt everything the checkpoint is supposed to restore.
        with torch.no_grad():
            for parameter in state.model.parameters():
                parameter.add_(torch.randn_like(parameter))
        for entry in state.optimizer.state.values():
            entry["exp_avg"].add_(1.0)
        state.step = 0
        state.tokens_seen = 0

        manager.load(state)

        weights_after = state.model.state_dict()
        assert all(
            torch.allclose(weights_before[k], weights_after[k]) for k in weights_before
        )
        assert torch.allclose(moment_before, state.optimizer.state[key]["exp_avg"])
        assert state.step == step_before

    def test_resumed_weights_and_rng_reproduce_the_next_step(self, tmp_path) -> None:
        """A resumed run must continue the trajectory, not restart it.

        The data cursor is deliberately held fixed here by feeding both runs the
        same batch: this test isolates the *weights plus RNG* half of resume.
        Loader-cursor resume is the data subsystem's own test, and mixing the
        two makes a failure impossible to attribute.
        """
        source, state, objective = build_stack(seed=5)
        run_steps(source, state, objective, 4)
        manager = CheckpointManager(tmp_path / "ckpt")
        manager.save(state.step, state)
        manager.wait()

        batch = next(iter(source))
        patchifier = GridPatchifier()
        _, uninterrupted = train_step(state, batch, objective, patchifier=patchifier)

        _, fresh_state, fresh_objective = build_stack(seed=5)
        manager.load(fresh_state)
        _, resumed = train_step(
            fresh_state, batch, fresh_objective, patchifier=patchifier
        )
        assert float(resumed.loss) == float(uninterrupted.loss)


class TestSimulatorAgreesWithTheRealModel:
    def test_model_shape_prices_correctly(self) -> None:
        model = VideoDiT(preset("tiny"))
        model.init_weights()
        shape = model.model_shape(sequence_length=512)
        assert shape.parameters == sum(p.numel() for p in model.parameters())
        report = simulate_config(shape, ParallelDims(world_size=8))
        assert report.fits
        assert report.to_dict()["parameters"] == shape.parameters

    def test_parallel_plan_applies_to_the_real_model(self) -> None:
        # The check that would otherwise cost a cluster allocation to make.
        dims = ParallelDims(world_size=64, context=4)
        with fake_world(dims, device_type="cpu") as world:
            model = VideoDiT(preset("tiny"))
            model.init_weights()
            total = sum(p.numel() for p in model.parameters())
            result = parallelize(model, dims, mesh=world.require_mesh())
            local = sum(
                p.to_local().numel() if hasattr(p, "to_local") else p.numel()
                for p in result.model.parameters()
            )
        assert any("fsdp" in item for item in result.applied)
        assert local < total

    def test_tensor_parallel_plan_is_declared_and_applies(self) -> None:
        dims = ParallelDims(world_size=4, tensor=4)
        with fake_world(dims, device_type="cpu") as world:
            model = VideoDiT(preset("tiny"))
            model.init_weights()
            result = parallelize(model, dims, mesh=world.require_mesh())
        assert any("tensor_parallel" in item for item in result.applied)


class TestConditioningTasks:
    def test_every_task_produces_a_finite_loss(self) -> None:
        from avgen.core import ConditionMode

        source, state, _ = build_stack()
        batch = next(iter(source))
        patchifier = GridPatchifier()
        for mode in (
            ConditionMode.JOINT,
            ConditionMode.IMAGE_TO_VIDEO,
            ConditionMode.CONTINUATION,
            ConditionMode.INPAINT,
        ):
            objective = FlowMatchingObjective(
                FlowMatchingConfig(),
                build_timestep_sampler("uniform"),
                MultiTaskConditioning(mode_weights=((mode, 1.0),)),
            )
            output = objective(
                state.model, batch, state.rng, patchifier=patchifier
            )
            loss = float(output.loss)
            assert loss == loss, f"{mode.name} produced a non-finite loss"
            assert loss > 0.0


class TestLoaderShutdown:
    """The prefetch thread must be gone before the interpreter tears down.

    This can only be observed from a separate process. A daemon thread still
    holding memory-mapped shard tensors at teardown makes the C++ runtime abort
    with "terminate called without an active exception" and exit code 134 — after
    a training run that actually succeeded. Every scheduler and CI system reads
    that as a failed job, so the exit code is the thing under test, not the
    output.
    """

    def test_early_break_leaves_no_live_thread(self, tmp_path) -> None:
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            f"""
            from pathlib import Path
            from avgen.data import (
                ConcatShardReader,
                ShardReader,
                SyntheticConfig,
                SyntheticSource,
                build_loader,
                write_shard,
            )

            shard = Path({str(tmp_path / "shard")!r})
            source = SyntheticSource(
                SyntheticConfig(
                    seed=0, num_samples=16, frames=4, height=4, width=4,
                    audio_frames=0, text_tokens=4, text_width=16,
                )
            )
            write_shard(shard, [source[i] for i in range(16)])
            store = ConcatShardReader([ShardReader(shard)])

            # Break while the producer is mid-put: this is what used to leave a
            # live daemon thread behind.
            for _ in build_loader(store, batch_size=2):
                break
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, (
            f"loader shutdown aborted the interpreter (exit {result.returncode}); "
            f"stderr: {result.stderr[-400:]}"
        )
        assert "terminate called" not in result.stderr
        assert "ok" in result.stdout
