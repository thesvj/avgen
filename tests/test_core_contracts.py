"""Tests for the frozen core contracts.

These tests exist to make the contracts hard to change by accident. Every
assertion here corresponds to a property some other subsystem relies on, so a
failure means something downstream is now silently wrong rather than merely
different.
"""

from __future__ import annotations

import pytest
import torch

from avgen.core import (
    ConditionMode,
    GridPatchifier,
    MediaBatch,
    MediaBatchSpec,
    ModelInput,
    ModelOutput,
    PatchLayout,
    RNGStreams,
    StepMetrics,
    TensorBundle,
    TensorBundleSpec,
    TensorDType,
    TextContext,
    TokenStream,
    build_temporal_coords,
    patchify_grid,
    stack_batches,
    unpatchify_grid,
)


def make_spec(batch: int = 2, audio_frames: int = 8) -> MediaBatchSpec:
    return MediaBatchSpec(
        schema_version=1,
        bucket_id=0,
        video_shape=(batch, 4, 4, 8, 8),
        audio_shape=(batch, 2, audio_frames),
        text_shape=(batch, 6, 16),
        video_timebase_num=25,
        video_timebase_den=1,
        audio_timebase_num=30000,
        audio_timebase_den=1001,
        video_codec_id="reference-video-v1",
        audio_codec_id="reference-audio-v1",
    )


def make_batch(spec: MediaBatchSpec) -> MediaBatch:
    batch, _, frames, height, width = spec.video_shape
    audio_frames = spec.audio_shape[2]
    return MediaBatch(
        video=torch.randn(spec.video_shape),
        audio=torch.randn(spec.audio_shape),
        text=torch.randn(spec.text_shape),
        video_mask=torch.ones((batch, frames, height, width), dtype=torch.bool),
        audio_mask=torch.ones((batch, audio_frames), dtype=torch.bool),
        video_positions=torch.arange(frames, dtype=torch.float32).repeat(batch, 1) / 25,
        audio_positions=torch.arange(audio_frames, dtype=torch.float32).repeat(
            batch, 1
        ),
        sample_ids=torch.arange(batch, dtype=torch.int64),
        spec=spec,
    )


class TestMediaBatch:
    def test_validate_accepts_a_well_formed_batch(self) -> None:
        make_batch(make_spec()).validate()

    def test_sample_ids_must_stay_on_cpu(self) -> None:
        # Moving sample_ids to the device would force a synchronisation every
        # time one is logged, which at 1000 ranks serialises the whole job.
        spec = make_spec()
        batch = make_batch(spec)
        assert batch.sample_ids.device.type == "cpu"

    def test_rejects_a_shape_mismatch(self) -> None:
        spec = make_spec()
        batch = make_batch(spec)
        broken = MediaBatch(
            video=torch.randn(2, 4, 4, 8, 9),  # width disagrees with the spec
            audio=batch.audio,
            text=batch.text,
            video_mask=batch.video_mask,
            audio_mask=batch.audio_mask,
            video_positions=batch.video_positions,
            audio_positions=batch.audio_positions,
            sample_ids=batch.sample_ids,
            spec=spec,
        )
        with pytest.raises(ValueError, match="video shape must be"):
            broken.validate()

    def test_exact_rational_timebases(self) -> None:
        # 30000/1001 is not representable in binary floating point; storing the
        # rational is what stops audio drifting out of sync over a long clip.
        spec = make_spec()
        assert spec.audio_timebase_num == 30000
        assert spec.audio_timebase_den == 1001
        assert spec.audio_fps == pytest.approx(29.97002997, rel=1e-9)

    def test_zero_audio_frames_is_legal(self) -> None:
        spec = make_spec(audio_frames=0)
        assert not spec.has_audio
        make_batch(spec).validate()

    def test_stack_batches_concatenates_and_rewrites_the_spec(self) -> None:
        spec = make_spec()
        stacked = stack_batches([make_batch(spec), make_batch(spec)])
        assert stacked.spec.batch_size == 4
        assert stacked.video.shape[0] == 4
        stacked.validate()

    def test_stack_batches_rejects_mismatched_specs(self) -> None:
        with pytest.raises(ValueError, match="identical MediaBatchSpec"):
            stack_batches([make_batch(make_spec()), make_batch(make_spec(batch=3))])

    def test_spec_round_trips_through_json(self) -> None:
        spec = make_spec()
        assert MediaBatchSpec.from_dict(spec.to_dict()) == spec


class TestPatchify:
    def test_round_trip_is_exact(self) -> None:
        layout = PatchLayout(frames=4, height=8, width=8, channels=4)
        grid = torch.randn(2, 4, 4, 8, 8)
        assert torch.equal(unpatchify_grid(patchify_grid(grid, layout), layout), grid)

    def test_token_count_matches_the_layout(self) -> None:
        layout = PatchLayout(frames=4, height=8, width=8, channels=4)
        assert layout.num_tokens == 4 * 4 * 4
        assert layout.patch_dim == 4 * 1 * 2 * 2

    def test_layout_rejects_indivisible_patching(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            PatchLayout(frames=4, height=7, width=8, patch_height=2)

    def test_coords_carry_seconds_not_indices(self) -> None:
        layout = PatchLayout(frames=4, height=4, width=4, channels=2)
        positions = torch.tensor([[0.0, 0.04, 0.08, 0.12]])
        coords = build_temporal_coords(positions, layout)
        assert coords.shape == (1, layout.num_tokens, 3)
        # Every token of the first temporal patch carries its physical time.
        first_frame_tokens = coords[0, : (layout.grid[1] * layout.grid[2]), 0]
        assert torch.allclose(first_frame_tokens, torch.zeros_like(first_frame_tokens))

    def test_patchifier_reduces_masks_correctly(self) -> None:
        # A patch containing any real latent is a real token; a patch is only a
        # clean anchor if every latent in it is clean.
        patchifier = GridPatchifier(patch_height=2, patch_width=2)
        grid = torch.randn(1, 2, 2, 4, 4)
        mask = torch.zeros((1, 2, 4, 4), dtype=torch.bool)
        mask[0, 0, 0, 0] = True
        conditioned = torch.ones((1, 2, 4, 4), dtype=torch.bool)
        conditioned[0, 0, 0, 0] = False
        stream = patchifier.to_tokens(
            grid,
            positions=torch.zeros(1, 2),
            mask=mask,
            noise_level=torch.zeros(1),
            conditioned=conditioned,
        )
        assert bool(stream.mask[0, 0]) is True
        assert bool(stream.conditioned[0, 0]) is False
        assert bool(stream.conditioned[0, 1]) is True

    def test_patchifier_round_trips_through_a_stream(self) -> None:
        patchifier = GridPatchifier()
        grid = torch.randn(2, 4, 4, 8, 8)
        stream = patchifier.to_tokens(
            grid,
            positions=torch.zeros(2, 4),
            mask=torch.ones((2, 4, 8, 8), dtype=torch.bool),
            noise_level=torch.zeros(2),
        )
        assert torch.equal(patchifier.to_grid(stream), grid)


class TestTokenStream:
    def test_loss_mask_excludes_padding_and_conditioning(self) -> None:
        stream = TokenStream(
            tokens=torch.randn(1, 4, 8),
            coords=torch.zeros(1, 4, 3),
            mask=torch.tensor([[True, True, True, False]]),
            noise_level=torch.zeros(1),
            layout=PatchLayout(frames=1, height=2, width=2, channels=8),
            conditioned=torch.tensor([[True, False, False, False]]),
        )
        assert stream.loss_mask().tolist() == [[False, True, True, False]]

    def test_per_sample_and_per_token_noise_both_validate(self) -> None:
        layout = PatchLayout(frames=1, height=2, width=2, channels=8)
        for noise in (torch.zeros(1), torch.zeros(1, 4)):
            TokenStream(
                tokens=torch.randn(1, 4, 8),
                coords=torch.zeros(1, 4, 3),
                mask=torch.ones((1, 4), dtype=torch.bool),
                noise_level=noise,
                layout=layout,
            ).validate()

    def test_empty_stream_stands_in_for_an_absent_modality(self) -> None:
        stream = TokenStream.empty_like(2, 16)
        stream.validate()
        assert stream.length == 0
        assert stream.is_empty


class TestRNGStreams:
    def test_streams_are_independent(self) -> None:
        # Drawing from the timestep stream must not perturb the noise stream, or
        # changing the timestep law silently changes the training noise.
        a = RNGStreams.from_seed(0)
        b = RNGStreams.from_seed(0)
        torch.randn(100, generator=b.timestep)
        assert torch.equal(
            torch.randn(4, generator=a.noise), torch.randn(4, generator=b.noise)
        )

    def test_data_rank_varies_but_only_on_data_rank(self) -> None:
        # This is the rule that makes tensor- and context-parallel ranks agree
        # about what sample they are denoising.
        first = RNGStreams.for_rank(0, data_rank=0)
        same = RNGStreams.for_rank(0, data_rank=0)
        other = RNGStreams.for_rank(0, data_rank=1)
        assert torch.equal(
            torch.randn(8, generator=first.noise), torch.randn(8, generator=same.noise)
        )
        assert not torch.equal(
            torch.randn(8, generator=RNGStreams.for_rank(0, data_rank=0).noise),
            torch.randn(8, generator=other.noise),
        )

    def test_state_round_trips(self) -> None:
        rng = RNGStreams.from_seed(7)
        state = rng.state_dict()
        expected = torch.randn(4, generator=rng.noise)
        rng.load_state_dict(state)
        assert torch.equal(torch.randn(4, generator=rng.noise), expected)

    def test_load_rejects_a_wrong_key_set(self) -> None:
        rng = RNGStreams.from_seed(0)
        state = rng.state_dict()
        del state["noise"]
        with pytest.raises(ValueError, match="must match named streams"):
            rng.load_state_dict(state)


class TestModelABI:
    def make_input(self, *, audio: bool = True) -> ModelInput:
        layout = PatchLayout(frames=2, height=4, width=4, channels=8)
        video = TokenStream(
            tokens=torch.randn(2, layout.num_tokens, layout.patch_dim),
            coords=torch.zeros(2, layout.num_tokens, 3),
            mask=torch.ones((2, layout.num_tokens), dtype=torch.bool),
            noise_level=torch.rand(2),
            layout=layout,
        )
        audio_stream = (
            TokenStream(
                tokens=torch.randn(2, 6, 4),
                coords=torch.zeros(2, 6, 3),
                mask=torch.ones((2, 6), dtype=torch.bool),
                noise_level=torch.rand(2),
                layout=PatchLayout.temporal(frames=6, channels=4),
            )
            if audio
            else TokenStream.empty_like(2, 4)
        )
        return ModelInput(
            video=video,
            audio=audio_stream,
            text=TextContext(
                features=torch.randn(2, 5, 16),
                mask=torch.ones((2, 5), dtype=torch.bool),
            ),
        )

    def test_validates_with_and_without_audio(self) -> None:
        self.make_input(audio=True).validate()
        self.make_input(audio=False).validate()

    def test_defaults_to_joint_conditioning(self) -> None:
        inputs = self.make_input()
        assert inputs.condition_mode.tolist() == [int(ConditionMode.JOINT)] * 2

    def test_unconditional_nulls_only_text(self) -> None:
        # Structural conditioning must survive: dropping it would make the null
        # branch a different task, and the guidance difference meaningless.
        inputs = self.make_input()
        null = inputs.unconditional()
        assert bool(null.text.features.abs().sum() == 0)
        assert torch.equal(null.video.tokens, inputs.video.tokens)

    def test_output_validation_catches_a_shape_mismatch(self) -> None:
        inputs = self.make_input()
        good = ModelOutput(
            video=torch.zeros_like(inputs.video.tokens),
            audio=torch.zeros_like(inputs.audio.tokens),
        )
        good.validate(inputs)
        bad = ModelOutput(
            video=torch.zeros(2, 3, 4), audio=torch.zeros_like(inputs.audio.tokens)
        )
        with pytest.raises(ValueError, match="video output shape"):
            bad.validate(inputs)


class TestTensorBundle:
    def test_validates_against_its_spec(self) -> None:
        spec = TensorBundleSpec(
            names=("flow",),
            semantics=("optical flow target",),
            shapes=((2, 3),),
            dtypes=(TensorDType.FP32,),
        )
        TensorBundle((torch.zeros(2, 3),)).validate(spec)
        with pytest.raises(TypeError, match="dtype must be"):
            TensorBundle((torch.zeros(2, 3, dtype=torch.int64),)).validate(spec)

    def test_named_lookup(self) -> None:
        spec = TensorBundleSpec(
            names=("a", "b"),
            semantics=("first", "second"),
            shapes=((1,), (1,)),
            dtypes=(TensorDType.FP32, TensorDType.FP32),
        )
        bundle = TensorBundle((torch.tensor([1.0]), torch.tensor([2.0])))
        assert bundle.get(spec, "b").item() == 2.0
        with pytest.raises(KeyError, match="unknown bundle field"):
            bundle.get(spec, "c")


class TestStepMetrics:
    def test_zeros_validate(self) -> None:
        StepMetrics.zeros().validate()

    def test_rejects_an_attached_tensor(self) -> None:
        metrics = StepMetrics.zeros()
        attached = StepMetrics(
            loss=torch.zeros((), requires_grad=True),
            video_loss=metrics.video_loss,
            audio_loss=metrics.audio_loss,
            valid_video_tokens=metrics.valid_video_tokens,
            valid_audio_tokens=metrics.valid_audio_tokens,
            grad_norm=metrics.grad_norm,
            nonfinite=metrics.nonfinite,
            skipped=metrics.skipped,
        )
        with pytest.raises(ValueError, match="must be detached"):
            attached.validate()
