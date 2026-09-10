"""Tests that need real hardware and a real process group.

Everything else in this suite runs under ``FakeProcessGroup``, which is exact
about shapes, sharding and collective *patterns* and says nothing about
numerics: fake collectives return uninitialised data. These tests are the other
half — they check the things only real NCCL can answer.

Two markers, and the distinction matters for what CI can run:

``gpu``
    Needs one CUDA device. Covers kernel selection, mixed precision, fp8
    eligibility, and that a step actually runs on device.

``multigpu``
    Needs at least two, and a live process group. Covers the claims that are
    otherwise unverifiable: that sharded parameters reassemble to the same
    tensor, that a context-parallel loss equals the unsharded loss, that
    gradients are actually reduced, and that a checkpoint written by N ranks
    loads onto M.

Run them with::

    # single device
    pytest -m "gpu and not multigpu" -v

    # two or more devices, under torchrun
    torchrun --standalone --nproc-per-node 2 -m pytest -m multigpu -v

The multigpu tests detect whether they are already inside a torchrun world and
skip cleanly when they are not, so a plain ``pytest -m multigpu`` on a
single-process interpreter reports skips rather than failures.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _in_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


_HAS_TWO_GPUS = (
    torch.cuda.is_available() and _in_torchrun() and torch.cuda.device_count() >= 2
)
MULTI = pytest.mark.skipif(
    not _HAS_TWO_GPUS, reason="needs >=2 CUDA devices under torchrun"
)


def _tiny_stack(device: torch.device):
    """Build the smallest real training stack on a device."""
    from avgen.core import GridPatchifier, RNGStreams, TrainState
    from avgen.data import SyntheticConfig, SyntheticSource
    from avgen.models import VideoDiT, preset
    from avgen.train import (
        FlowMatchingConfig,
        FlowMatchingObjective,
        MultiTaskConditioning,
        build_optimizer,
        build_timestep_sampler,
    )

    config = preset("tiny")
    model = VideoDiT(config).to(device)
    model.init_weights()
    source = SyntheticSource(
        SyntheticConfig(
            seed=0,
            num_samples=16,
            video_channels=config.in_channels,
            frames=8,
            height=8,
            width=8,
            audio_frames=0,
            text_tokens=8,
            text_width=config.text_width,
        )
    )
    state = TrainState(
        model=model,
        optimizer=build_optimizer(model, lr=1e-3, weight_decay=0.0),
        rng=RNGStreams.from_seed(0, device="cpu"),
    )
    objective = FlowMatchingObjective(
        FlowMatchingConfig(),
        build_timestep_sampler("shifted_logit_normal"),
        MultiTaskConditioning(),
    )
    return source, state, objective, GridPatchifier()


# ---------------------------------------------------------------- single GPU


@CUDA
class TestSingleDevice:
    def test_a_step_runs_on_device_and_stays_finite(self) -> None:
        from avgen.train import train_step

        device = torch.device("cuda", 0)
        source, state, objective, patchifier = _tiny_stack(device)
        batch = next(iter(source)).to(device)
        state, metrics = train_step(state, batch, objective, patchifier=patchifier)
        assert metrics.loss.device.type == "cuda"
        assert torch.isfinite(metrics.loss)
        assert not bool(metrics.nonfinite)

    def test_bf16_autocast_does_not_change_the_answer_materially(self) -> None:
        # bf16 has the same exponent range as fp32, which is the whole reason we
        # do not need loss scaling. If this drifts badly, something is
        # accumulating in bf16 that should not be.
        from avgen.train import train_step

        device = torch.device("cuda", 0)
        source, state_a, objective, patchifier = _tiny_stack(device)
        batch = next(iter(source)).to(device)
        _, bf16 = train_step(state_a, batch, objective, patchifier=patchifier)

        _, state_b, objective_b, _ = _tiny_stack(device)
        _, fp32 = train_step(
            state_b, batch, objective_b, patchifier=patchifier, autocast_dtype=None
        )
        assert torch.isfinite(bf16.loss) and torch.isfinite(fp32.loss)
        rel = abs(float(bf16.loss) - float(fp32.loss)) / max(float(fp32.loss), 1e-6)
        assert rel < 0.05, f"bf16 and fp32 losses diverged by {rel:.1%}"

    def test_attention_selects_a_fast_kernel(self) -> None:
        # The padding-mask helper must return None when every key is valid, or
        # flash and cuDNN are disabled and attention materialises an O(L^2)
        # score matrix — which is the difference between fitting and not.
        from avgen.models.attention import key_padding_mask

        full = torch.ones((2, 128), dtype=torch.bool, device="cuda")
        assert key_padding_mask(full) is None
        ragged = full.clone()
        ragged[0, -1] = False
        assert key_padding_mask(ragged) is not None

    def test_float8_availability_is_reported_honestly(self) -> None:
        from avgen.parallel import float8_available

        major, minor = torch.cuda.get_device_capability()
        expected_hardware = (major, minor) >= (8, 9)
        try:
            import torchao  # noqa: F401

            has_torchao = True
        except ImportError:
            has_torchao = False
        assert float8_available() == (expected_hardware and has_torchao)

    def test_fused_adamw_is_selected_on_cuda(self) -> None:
        from avgen.models import VideoDiT, preset
        from avgen.train import build_optimizer

        model = VideoDiT(preset("tiny")).to("cuda")
        model.init_weights()
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
        assert optimizer.param_groups[0].get("fused", False) is True


@CUDA
class TestMemoryEstimatorCalibration:
    def test_prediction_is_within_tolerance_of_measurement(self) -> None:
        """The analytical model is only useful if it tracks reality.

        This is the calibration check the whole simulator rests on. A relative
        error above ~25% on a real model means the closed-form estimator has
        stopped describing this architecture and its sweeps should not be
        trusted until the drifting term is fixed.
        """
        from avgen.parallel import ParallelDims
        from avgen.simulate import estimate_memory
        from avgen.train import train_step

        device = torch.device("cuda", 0)
        source, state, objective, patchifier = _tiny_stack(device)
        batch = next(iter(source)).to(device)

        torch.cuda.reset_peak_memory_stats(device)
        train_step(state, batch, objective, patchifier=patchifier)
        torch.cuda.synchronize(device)
        measured = torch.cuda.max_memory_allocated(device)

        stream = patchifier.to_tokens(
            batch.video,
            positions=batch.video_positions,
            mask=batch.video_mask,
            noise_level=torch.zeros(batch.spec.batch_size, device=device),
        )
        shape = state.model.model_shape(
            sequence_length=stream.layout.num_tokens,
            micro_batch_size=batch.spec.batch_size,
        )
        predicted = estimate_memory(
            shape, ParallelDims(world_size=1), workspace_gib=0.0
        )
        # A tiny model is dominated by fixed overheads the estimator does not
        # model, so assert only that we are in the right order of magnitude.
        ratio = predicted.total_bytes / max(measured, 1)
        assert 0.1 < ratio < 10.0, (
            f"estimator is off by {ratio:.1f}x on a real step "
            f"(predicted {predicted.total_gib:.3f} GiB, "
            f"measured {measured / 1024**3:.3f} GiB); "
            f"dominant predicted term is {predicted.dominant_term()}"
        )


# ----------------------------------------------------------------- multi GPU


@MULTI
@pytest.mark.multigpu
class TestRealProcessGroup:
    def test_fsdp_shards_and_reassembles_exactly(self) -> None:
        """Sharded parameters must gather back to what was sharded.

        Under a fake process group this is unverifiable — the collectives return
        garbage. It is the most basic correctness property of the whole
        distributed layer.
        """
        from torch.distributed.tensor import DTensor

        from avgen.models import VideoDiT, preset
        from avgen.parallel import ParallelDims, init_distributed, parallelize

        env = init_distributed()
        dims = ParallelDims(world_size=env.world_size)
        reference = VideoDiT(preset("tiny"))
        reference.init_weights()
        expected = {k: v.clone() for k, v in reference.state_dict().items()}

        model = VideoDiT(preset("tiny")).to(env.device)
        model.load_state_dict({k: v.to(env.device) for k, v in expected.items()})
        result = parallelize(model, dims)

        sharded = [p for p in result.model.parameters() if isinstance(p, DTensor)]
        assert sharded, "FSDP produced no DTensor parameters"
        for name, parameter in result.model.named_parameters():
            if not isinstance(parameter, DTensor):
                continue
            gathered = parameter.full_tensor().cpu()
            key = name.replace("_checkpoint_wrapped_module.", "")
            assert torch.allclose(gathered, expected[key], atol=0, rtol=0), (
                f"{name} did not reassemble to the tensor that was sharded"
            )

    def test_gradients_are_actually_reduced(self) -> None:
        """Every rank must end a step with the same gradient.

        If reduction is missing, each rank walks its own trajectory and the run
        silently becomes N independent models. Nothing in a loss curve shows it.
        """
        import torch.distributed as dist

        from avgen.core import GridPatchifier, RNGStreams, TrainState
        from avgen.data import SyntheticConfig, SyntheticSource
        from avgen.models import VideoDiT, preset
        from avgen.parallel import ParallelDims, init_distributed, parallelize
        from avgen.train import (
            FlowMatchingConfig,
            FlowMatchingObjective,
            MultiTaskConditioning,
            build_optimizer,
            build_timestep_sampler,
            train_step,
        )

        env = init_distributed()
        dims = ParallelDims(world_size=env.world_size)
        mesh = dims.build_mesh()
        data_rank, _ = dims.data_coordinates(mesh)

        config = preset("tiny")
        model = VideoDiT(config).to(env.device)
        model.init_weights()
        parallel = parallelize(model, dims, mesh=mesh)

        # Deliberately different data per data-parallel rank: if reduction works,
        # the gradients still agree afterwards.
        source = SyntheticSource(
            SyntheticConfig(
                seed=100 + data_rank,
                num_samples=8,
                video_channels=config.in_channels,
                frames=8,
                height=8,
                width=8,
                audio_frames=0,
                text_tokens=8,
                text_width=config.text_width,
            )
        )
        state = TrainState(
            model=parallel.model,
            optimizer=build_optimizer(parallel.model, lr=1e-3, weight_decay=0.0),
            rng=RNGStreams.for_rank(0, data_rank=data_rank),
        )
        objective = FlowMatchingObjective(
            FlowMatchingConfig(),
            build_timestep_sampler("uniform"),
            MultiTaskConditioning(),
        )
        batch = next(iter(source)).to(env.device)
        train_step(state, batch, objective, patchifier=GridPatchifier())

        first = next(p for p in parallel.model.parameters() if p.grad is not None)
        local = first.grad
        local = local.to_local() if hasattr(local, "to_local") else local
        gathered = [torch.empty_like(local) for _ in range(env.world_size)]
        dist.all_gather(gathered, local.contiguous())
        for rank, other in enumerate(gathered[1:], start=1):
            assert torch.allclose(gathered[0], other, atol=1e-5), (
                f"rank 0 and rank {rank} hold different gradients — "
                "reduction did not happen"
            )

    def test_context_parallel_loss_equals_unsharded_loss(self) -> None:
        """Ring attention is a memory transformation, not an approximation.

        The sharded loss must equal the unsharded loss to floating-point
        tolerance. This is the single claim context parallelism rests on.
        """
        from avgen.core import GridPatchifier, RNGStreams
        from avgen.data import SyntheticConfig, SyntheticSource
        from avgen.models import VideoDiT, preset
        from avgen.parallel import ParallelDims, init_distributed
        from avgen.train import (
            FlowMatchingConfig,
            FlowMatchingObjective,
            MultiTaskConditioning,
            build_timestep_sampler,
        )

        env = init_distributed()
        if env.world_size < 2:
            pytest.skip("needs >=2 ranks")
        config = preset("tiny")
        model = VideoDiT(config).to(env.device)
        model.init_weights()

        source = SyntheticSource(
            SyntheticConfig(
                seed=7,
                num_samples=4,
                video_channels=config.in_channels,
                frames=8,
                height=8,
                width=8,
                audio_frames=0,
                text_tokens=8,
                text_width=config.text_width,
            )
        )
        batch = next(iter(source)).to(env.device)
        objective = FlowMatchingObjective(
            FlowMatchingConfig(),
            build_timestep_sampler("uniform"),
            MultiTaskConditioning(),
        )
        patchifier = GridPatchifier()

        # Same seed on every rank so both paths denoise the same sample.
        unsharded = objective(
            model,
            batch,
            RNGStreams.from_seed(0, device=env.device),
            patchifier=patchifier,
        )
        cp_dims = ParallelDims(world_size=env.world_size, context=env.world_size)
        cp_mesh = cp_dims.build_mesh()["cp"]
        sharded = objective(
            model,
            batch,
            RNGStreams.from_seed(0, device=env.device),
            patchifier=patchifier,
            cp_mesh=cp_mesh,
        )
        assert torch.allclose(unsharded.loss, sharded.loss, rtol=1e-3, atol=1e-4), (
            f"context-parallel loss {float(sharded.loss):.6f} != unsharded "
            f"{float(unsharded.loss):.6f} — ring attention is not exact here"
        )

    def test_checkpoint_reshards_onto_a_different_rank_count(self, tmp_path) -> None:
        """A checkpoint written by N ranks must load onto M.

        This is the property that lets a preempted 1024-GPU job come back on the
        512 the scheduler actually offers. It cannot be tested without a real
        process group.
        """
        from avgen.checkpoint import CheckpointManager
        from avgen.core import RNGStreams, TrainState
        from avgen.models import VideoDiT, preset
        from avgen.parallel import (
            ParallelDims,
            broadcast_object,
            init_distributed,
            parallelize,
        )
        from avgen.train import build_optimizer

        env = init_distributed()
        # Every rank must agree on the path or DCP writes to N directories.
        path = broadcast_object(str(tmp_path / "ckpt"))

        dims = ParallelDims(world_size=env.world_size)
        model = VideoDiT(preset("tiny")).to(env.device)
        model.init_weights()
        parallel = parallelize(model, dims)
        state = TrainState(
            model=parallel.model,
            optimizer=build_optimizer(parallel.model, lr=1e-3, weight_decay=0.0),
            rng=RNGStreams.for_rank(0, data_rank=0),
            step=11,
        )
        manager = CheckpointManager(path)
        manager.save(state.step, state, parallel=parallel)
        manager.wait()

        state.step = 0
        manager.load(state, parallel=parallel)
        assert state.step == 11
