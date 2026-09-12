"""Tests for the data subsystem: shards, the loader, bucketing, and packing.

These check the four properties the rest of avgen silently depends on and that
nothing downstream can detect when they break:

* a shard read back is **byte-identical** to the shard written, and a shard that
  rotted by one bit is **refused** rather than trained on;
* two ranks sharing a ``data_rank`` see identical batches, and two ranks that do
  not see disjoint ones covering the corpus exactly;
* a resumed cursor lands on the same sample, and a resume that would silently
  change the sample order is refused;
* a packed sequence's block-diagonal mask matches its segment ids exactly, so no
  clip ever attends to another clip's tokens.

All of it is CPU-only at tiny shapes.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from avgen.core.tokens import PatchLayout, TokenStream
from avgen.data import (
    Bucket,
    BucketCurriculum,
    BucketPlan,
    BucketSampler,
    ConcatShardReader,
    CurriculumPhase,
    LoaderCursor,
    ShardCorruptionError,
    ShardManifest,
    ShardReader,
    ShardRecord,
    SyntheticConfig,
    SyntheticSource,
    assign_buckets,
    build_loader,
    largest_remainder,
    pack_streams,
    plan_packing,
    split_packed,
    validate_shard,
    write_shard,
)
from avgen.data.packing import PADDING_SEGMENT
from avgen.data.shard import COMMIT_FILENAME, DATA_FILENAME, MANIFEST_FILENAME

# Deliberately smaller than the SyntheticConfig defaults: every test here writes
# real files, and a shard is a directory whose container is the sum of its
# samples.
TINY = {
    "frames": 8,
    "height": 4,
    "width": 4,
    "video_channels": 2,
    "text_tokens": 4,
    "text_width": 8,
}


def make_source(
    *, num_samples: int = 16, seed: int = 0, **overrides
) -> SyntheticSource:
    settings = {**TINY, "num_samples": num_samples, "seed": seed, **overrides}
    return SyntheticSource(SyntheticConfig(**settings))


def make_shard(path: Path, *, num_samples: int = 8, **overrides) -> Path:
    """Write one committed shard directory and return it."""
    source = make_source(num_samples=num_samples, **overrides)
    write_shard(path, [source[index] for index in range(num_samples)])
    return path


def flip_one_byte(path: Path, offset: int = -1) -> None:
    """Flip a single bit of a file in place, preserving its length."""
    payload = bytearray(path.read_bytes())
    position = offset if offset >= 0 else len(payload) + offset
    payload[position] ^= 0x01
    path.write_bytes(bytes(payload))


class TestShardRoundTrip:
    def test_every_tensor_is_bit_identical_after_a_round_trip(self, tmp_path) -> None:
        # Not "close": a latent that changed by one ulp on the way to disk is a
        # corrupt shard, and approximate comparison here would hide it.
        source = make_source(num_samples=6)
        written = [source[index] for index in range(6)]
        write_shard(tmp_path / "shard", written)

        with ShardReader(tmp_path / "shard") as reader:
            assert len(reader) == 6
            for index, original in enumerate(written):
                restored = reader[index]
                assert restored.descriptor == original.descriptor
                for name in (
                    "video",
                    "audio",
                    "text",
                    "text_mask",
                    "video_positions",
                    "audio_positions",
                ):
                    left = getattr(original, name)
                    right = getattr(restored, name)
                    assert right.dtype is left.dtype, name
                    assert torch.equal(right, left), f"{name} differs at sample {index}"

    def test_a_video_only_corpus_round_trips_its_empty_tensors(self, tmp_path) -> None:
        # Zero-element tensors are not stored in the container at all; they are
        # rebuilt from the manifest. That reconstruction is the part worth
        # checking, because an off-by-one there is invisible until collation.
        source = make_source(num_samples=2, audio_frames=0, text_tokens=0)
        write_shard(tmp_path / "shard", [source[index] for index in range(2)])
        with ShardReader(tmp_path / "shard") as reader:
            sample = reader[0]
            assert sample.audio.shape[1] == 0
            assert sample.text.shape[0] == 0
            assert sample.text_mask.dtype is torch.bool
            sample.validate()

    def test_unknown_manifest_keys_survive_the_round_trip(self, tmp_path) -> None:
        # A shard outlives the version of avgen that wrote it; dropping a key a
        # newer writer recorded would destroy information on rewrite.
        extra = {"pipeline_version": "7.2", "licence": "CC-BY", "shots": [1, 2, 3]}
        make_shard(tmp_path / "shard", num_samples=2)
        # write_shard refuses to rewrite a committed shard, so write a second.
        source = make_source(num_samples=2)
        manifest = write_shard(
            tmp_path / "extra-shard",
            [source[index] for index in range(2)],
            extra=extra,
        )
        assert dict(manifest.extra) == extra
        with ShardReader(tmp_path / "extra-shard") as reader:
            assert dict(reader.manifest.extra) == extra

    def test_unknown_record_keys_survive_a_manifest_round_trip(self) -> None:
        source = make_source(num_samples=1)
        record = ShardRecord(descriptor=source.descriptor(0))
        values = {**record.to_dict(), "quality_score": 0.87}
        restored = ShardRecord.from_dict(values)
        assert restored.extra == {"quality_score": 0.87}
        assert restored.to_dict()["quality_score"] == 0.87

    def test_the_manifest_encoding_is_a_pure_function_of_its_content(
        self, tmp_path
    ) -> None:
        # Two writers producing identical samples must produce identical bytes,
        # or a digest cannot be used to compare two copies of a corpus.
        first = make_shard(tmp_path / "a", num_samples=3)
        second = make_shard(tmp_path / "b", num_samples=3)
        assert (first / DATA_FILENAME).read_bytes() == (
            second / DATA_FILENAME
        ).read_bytes()
        left = json.loads((first / MANIFEST_FILENAME).read_text())
        right = json.loads((second / MANIFEST_FILENAME).read_text())
        for payload in (left, right):
            payload.pop("created_unix")
        assert left == right


class TestShardValidation:
    def test_validate_passes_on_a_good_shard(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=4)
        manifest = validate_shard(shard, check_data=True)
        assert len(manifest) == 4
        assert manifest.data_bytes == (shard / DATA_FILENAME).stat().st_size

    def test_validate_detects_a_single_flipped_byte(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=4)
        before = (shard / DATA_FILENAME).stat().st_size
        flip_one_byte(shard / DATA_FILENAME)
        # The length is unchanged, so nothing but the digest can catch this.
        assert (shard / DATA_FILENAME).stat().st_size == before
        with pytest.raises(ShardCorruptionError, match="data digest mismatch"):
            validate_shard(shard, check_data=True)

    def test_a_flipped_byte_slips_past_the_cheap_check(self, tmp_path) -> None:
        # Documents the cost of the default: opening a shard does not re-digest
        # it, so bit rot is only caught by the offline sweep. If this ever
        # starts raising, the loader just got much more expensive to start.
        shard = make_shard(tmp_path / "shard", num_samples=4)
        flip_one_byte(shard / DATA_FILENAME)
        validate_shard(shard, check_data=False)

    def test_an_edited_manifest_is_caught_by_the_commit_marker(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=4)
        payload = json.loads((shard / MANIFEST_FILENAME).read_text())
        payload["data_bytes"] = payload["data_bytes"] + 1
        (shard / MANIFEST_FILENAME).write_text(json.dumps(payload))
        with pytest.raises(ShardCorruptionError, match="manifest digest mismatch"):
            validate_shard(shard, check_data=False)

    def test_a_truncated_container_is_caught_without_digesting(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=4)
        data = shard / DATA_FILENAME
        data.write_bytes(data.read_bytes()[:-64])
        with pytest.raises(ShardCorruptionError, match="truncated or grew"):
            validate_shard(shard, check_data=False)

    def test_an_uncommitted_shard_is_refused(self, tmp_path) -> None:
        # The preempted-writer case. The files are all there and all correct;
        # the absence of the marker is the whole signal.
        shard = make_shard(tmp_path / "shard", num_samples=4)
        (shard / COMMIT_FILENAME).unlink()
        with pytest.raises(ShardCorruptionError, match="never committed"):
            validate_shard(shard)
        with pytest.raises(ShardCorruptionError, match="never committed"):
            ShardReader(shard)

    def test_write_shard_refuses_to_overwrite_a_committed_shard(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=2)
        source = make_source(num_samples=2)
        with pytest.raises(FileExistsError, match="immutable"):
            write_shard(shard, [source[0]])

    def test_write_shard_requires_at_least_one_sample(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            write_shard(tmp_path / "shard", [])

    def test_a_manifest_that_miscounts_its_samples_is_corrupt(self) -> None:
        source = make_source(num_samples=1)
        manifest = ShardManifest(
            shard_schema_version=1,
            data_schema_version=1,
            records=(ShardRecord(descriptor=source.descriptor(0)),),
            data_sha256="0" * 64,
            data_bytes=0,
            created_unix=0,
        )
        values = manifest.to_dict()
        values["sample_count"] = 5
        with pytest.raises(ShardCorruptionError, match="declares 5 samples"):
            ShardManifest.from_dict(values)


class TestConcatShardReader:
    def test_global_indices_resolve_to_the_right_shard(self, tmp_path) -> None:
        make_shard(tmp_path / "s0", num_samples=3)
        make_shard(tmp_path / "s1", num_samples=5)
        with ConcatShardReader.from_paths([tmp_path / "s0", tmp_path / "s1"]) as corpus:
            assert len(corpus) == 8
            assert corpus.locate(0) == (0, 0)
            assert corpus.locate(2) == (0, 2)
            assert corpus.locate(3) == (1, 0)
            assert corpus.locate(7) == (1, 4)
            with pytest.raises(IndexError):
                corpus.locate(8)

    def test_uncommitted_shards_can_be_skipped_but_are_never_silent(
        self, tmp_path
    ) -> None:
        make_shard(tmp_path / "s0", num_samples=3)
        make_shard(tmp_path / "s1", num_samples=3)
        (tmp_path / "s1" / COMMIT_FILENAME).unlink()
        paths = [tmp_path / "s0", tmp_path / "s1"]
        with pytest.raises(ShardCorruptionError):
            ConcatShardReader.from_paths(paths)
        with ConcatShardReader.from_paths(paths, skip_uncommitted=True) as corpus:
            assert len(corpus) == 3

    def test_a_corpus_must_not_mix_codecs(self, tmp_path) -> None:
        make_shard(tmp_path / "s0", num_samples=2)
        make_shard(tmp_path / "s1", num_samples=2, video_codec_id="other-vae-v9")
        readers = [ShardReader(tmp_path / "s0"), ShardReader(tmp_path / "s1")]
        try:
            with pytest.raises(ValueError, match="same codecs and timebases"):
                ConcatShardReader(readers)
        finally:
            for reader in readers:
                reader.close()


class TestLoaderDeterminism:
    def test_the_same_data_rank_yields_byte_identical_batches(self, tmp_path) -> None:
        # Context- and tensor-parallel ranks share a data_rank and hold shards
        # of the SAME sample: if their batches differ the gradient is wrong and
        # nothing downstream can tell.
        shard = make_shard(tmp_path / "shard", num_samples=16)
        streams = []
        for _ in range(2):
            with ConcatShardReader.from_paths([shard]) as corpus:
                loader = build_loader(
                    corpus,
                    batch_size=2,
                    seed=7,
                    data_rank=1,
                    data_world=2,
                    max_epochs=1,
                    pin_memory=False,
                    prefetch=False,
                )
                streams.append(
                    [
                        (batch.sample_ids.clone(), batch.video.clone())
                        for batch in loader
                    ]
                )
        first, second = streams
        assert len(first) == 4
        for (left_ids, left_video), (right_ids, right_video) in zip(
            first, second, strict=True
        ):
            assert torch.equal(left_ids, right_ids)
            assert torch.equal(left_video, right_video)

    def test_different_data_ranks_partition_the_corpus_exactly(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=16)
        seen: list[list[int]] = []
        for rank in range(2):
            with ConcatShardReader.from_paths([shard]) as corpus:
                loader = build_loader(
                    corpus,
                    batch_size=2,
                    seed=7,
                    data_rank=rank,
                    data_world=2,
                    max_epochs=1,
                    pin_memory=False,
                    prefetch=False,
                )
                seen.append(
                    [int(value) for batch in loader for value in batch.sample_ids]
                )
        left, right = (set(item) for item in seen)
        assert left.isdisjoint(right), "two data ranks trained on the same samples"
        assert left | right == set(range(16)), "the epoch did not cover the corpus"
        assert len(seen[0]) == len(seen[1]) == 8

    def test_every_rank_reports_the_same_epoch_length(self, tmp_path) -> None:
        # Ranks that disagree here hang the job at the end-of-epoch collective,
        # minutes after the rank that ran out went quiet.
        shard = make_shard(tmp_path / "shard", num_samples=14)
        with ConcatShardReader.from_paths([shard]) as corpus:
            lengths = {
                build_loader(
                    corpus,
                    batch_size=2,
                    data_rank=rank,
                    data_world=3,
                    pin_memory=False,
                    prefetch=False,
                ).batches_per_epoch()
                for rank in range(3)
            }
        assert lengths == {2}

    def test_a_corpus_too_small_for_one_batch_per_rank_is_refused(
        self, tmp_path
    ) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=4)
        with (
            ConcatShardReader.from_paths([shard]) as corpus,
            pytest.raises(ValueError, match="every rank must"),
        ):
            build_loader(corpus, batch_size=2, data_rank=0, data_world=8)


class TestLoaderResume:
    def build(self, corpus, **overrides):
        settings = {
            "batch_size": 2,
            "seed": 3,
            "max_epochs": 2,
            "pin_memory": False,
            "prefetch": False,
        }
        settings.update(overrides)
        return build_loader(corpus, **settings)

    def test_resume_lands_on_the_identical_sample(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=12)
        with ConcatShardReader.from_paths([shard]) as corpus:
            reference = self.build(corpus)
            iterator = iter(reference)
            for _ in range(3):
                next(iterator)
            state = reference.state_dict()
            tail = [int(value) for value in next(iterator).sample_ids]

            resumed = self.build(corpus)
            resumed.load_state_dict(state)
            restored = [int(value) for value in next(iter(resumed)).sample_ids]
        assert restored == tail
        assert state["cursor"]["batch_index"] == 3
        assert state["cursor"]["samples_seen"] == 6

    def test_resume_refuses_a_changed_seed(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=12)
        with ConcatShardReader.from_paths([shard]) as corpus:
            state = self.build(corpus).state_dict()
            other = self.build(corpus, seed=99)
            with pytest.raises(ValueError, match="seed"):
                other.load_state_dict(state)

    def test_resume_refuses_a_changed_corpus(self, tmp_path) -> None:
        make_shard(tmp_path / "s0", num_samples=12)
        make_shard(tmp_path / "s1", num_samples=12)
        with ConcatShardReader.from_paths([tmp_path / "s0"]) as small:
            state = self.build(small).state_dict()
        with (
            ConcatShardReader.from_paths([tmp_path / "s0", tmp_path / "s1"]) as grown,
            pytest.raises(ValueError, match="the corpus changed"),
        ):
            self.build(grown).load_state_dict(state)

    def test_resume_refuses_a_reshard_unless_it_was_asked_for(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=16)
        with ConcatShardReader.from_paths([shard]) as corpus:
            state = self.build(corpus).state_dict()
            state["cursor"]["batch_index"] = 2
            with pytest.raises(ValueError, match="allow_reshard"):
                self.build(corpus, data_rank=0, data_world=2).load_state_dict(state)

            permitted = self.build(
                corpus, data_rank=0, data_world=2, allow_reshard=True
            )
            permitted.load_state_dict(state)
        # An approximate resume restarts the epoch rather than pretending to
        # land on a sample the new slicing does not contain.
        assert permitted.cursor.batch_index == 0

    def test_cursor_state_is_plain_integers(self) -> None:
        cursor = LoaderCursor(epoch=1, batch_index=2, samples_seen=30, step=7)
        assert cursor.state_dict() == {
            "epoch": 1,
            "batch_index": 2,
            "samples_seen": 30,
            "step": 7,
        }
        restored = LoaderCursor()
        restored.load_state_dict(cursor.state_dict())
        assert restored.state_dict() == cursor.state_dict()
        assert "samples_seen=30" in repr(cursor)
        with pytest.raises(KeyError, match="step"):
            LoaderCursor().load_state_dict(
                {"epoch": 0, "batch_index": 0, "samples_seen": 0}
            )

    def test_build_loader_rejects_two_sources_of_truth(self, tmp_path) -> None:
        shard = make_shard(tmp_path / "shard", num_samples=8)
        with (
            ConcatShardReader.from_paths([shard]) as corpus,
            pytest.raises(ValueError, match="must be given together"),
        ):
            build_loader(corpus, batch_size=2, dims=object())

    def test_assign_buckets_refuses_a_sample_that_matches_no_bucket(self) -> None:
        store = make_source(num_samples=4)
        plan = BucketPlan(buckets=(Bucket(bucket_id=0, frames=8, height=8, width=8),))
        with pytest.raises(ValueError, match="matches"):
            assign_buckets(store, plan)


class TestBucketing:
    def test_largest_remainder_sums_exactly(self) -> None:
        for total in (0, 1, 7, 10, 101):
            parts = largest_remainder(total, [0.2, 0.3, 0.5])
            assert sum(parts) == total
            assert len(parts) == 3
            assert all(part >= 0 for part in parts)

    def test_largest_remainder_breaks_ties_on_index(self) -> None:
        # Two ranks that disagree about which bucket got the spare batch produce
        # different batch counts and deadlock at the next collective.
        assert largest_remainder(1, [1.0, 1.0, 1.0]) == (1, 0, 0)
        assert largest_remainder(2, [1.0, 1.0, 1.0]) == (1, 1, 0)
        assert largest_remainder(4, [1.0, 1.0]) == (2, 2)

    def test_largest_remainder_rejects_a_degenerate_mixture(self) -> None:
        with pytest.raises(ValueError, match="at least one weight"):
            largest_remainder(4, [])
        with pytest.raises(ValueError, match="positive weight"):
            largest_remainder(4, [0.0, 0.0])
        with pytest.raises(ValueError, match="finite and non-negative"):
            largest_remainder(4, [1.0, -1.0])

    def test_a_curriculum_ramps_the_mixture_between_keyframes(self) -> None:
        curriculum = BucketCurriculum(
            phases=(
                CurriculumPhase(start_step=0, weights=(1.0, 0.0)),
                CurriculumPhase(start_step=100, weights=(0.0, 1.0)),
            )
        )
        assert curriculum.bucket_count == 2
        assert curriculum.weights_at(0) == pytest.approx((1.0, 0.0))
        assert curriculum.weights_at(50) == pytest.approx((0.5, 0.5))
        assert curriculum.weights_at(100) == pytest.approx((0.0, 1.0))
        # Clamped, not extrapolated: a linear extrapolation goes negative and a
        # negative mixture weight is not a thing.
        assert curriculum.weights_at(10_000) == pytest.approx((0.0, 1.0))

    def test_a_step_curriculum_holds_each_keyframe(self) -> None:
        curriculum = BucketCurriculum(
            phases=(
                CurriculumPhase(start_step=0, weights=(1.0, 0.0)),
                CurriculumPhase(start_step=100, weights=(0.0, 1.0)),
            ),
            interpolate=False,
        )
        assert curriculum.weights_at(99) == pytest.approx((1.0, 0.0))

    def test_curriculum_phases_are_sorted_and_validated(self) -> None:
        curriculum = BucketCurriculum(
            phases=(
                CurriculumPhase(start_step=10, weights=(1.0, 1.0)),
                CurriculumPhase(start_step=0, weights=(1.0, 0.0)),
            )
        )
        assert [phase.start_step for phase in curriculum.phases] == [0, 10]
        with pytest.raises(ValueError, match="distinct steps"):
            BucketCurriculum(
                phases=(
                    CurriculumPhase(start_step=0, weights=(1.0,)),
                    CurriculumPhase(start_step=0, weights=(0.5,)),
                )
            )
        with pytest.raises(ValueError, match="one weight per bucket"):
            BucketCurriculum(
                phases=(
                    CurriculumPhase(start_step=0, weights=(1.0,)),
                    CurriculumPhase(start_step=1, weights=(1.0, 1.0)),
                )
            )

    def two_bucket_sampler(self, **overrides) -> BucketSampler:
        plan = BucketPlan(
            buckets=(
                Bucket(bucket_id=0, frames=4, height=4, width=4),
                Bucket(bucket_id=1, frames=8, height=4, width=4),
            )
        )
        assignments = [0] * 16 + [1] * 16
        settings = {"batch_size": 2, "seed": 5}
        settings.update(overrides)
        return BucketSampler(plan, assignments, **settings)

    def test_the_curriculum_shifts_the_mixture_over_steps(self) -> None:
        curriculum = BucketCurriculum(
            phases=(
                CurriculumPhase(start_step=0, weights=(1.0, 0.0)),
                CurriculumPhase(start_step=100, weights=(0.0, 1.0)),
            )
        )
        sampler = self.two_bucket_sampler(curriculum=curriculum)
        assert sampler.quotas_at(0) == (16, 0)
        assert sampler.quotas_at(50) == (8, 8)
        assert sampler.quotas_at(100) == (0, 16)

        early = sampler.plan_epoch(epoch=0, step=0)
        late = sampler.plan_epoch(epoch=0, step=100)
        assert {batch.bucket_id for batch in early} == {0}
        assert {batch.bucket_id for batch in late} == {1}

    def test_the_epoch_length_does_not_move_with_the_mixture(self) -> None:
        # If it did, the learning-rate schedule would stretch and compress as
        # the curriculum ran.
        curriculum = BucketCurriculum(
            phases=(
                CurriculumPhase(start_step=0, weights=(1.0, 0.0)),
                CurriculumPhase(start_step=100, weights=(0.0, 1.0)),
            )
        )
        sampler = self.two_bucket_sampler(curriculum=curriculum)
        assert sampler.batches_per_epoch() == 16
        assert len(sampler.plan_epoch(epoch=0, step=0)) == 16
        assert len(sampler.plan_epoch(epoch=0, step=100)) == 16

    def test_an_epoch_plan_is_a_pure_function_of_seed_epoch_and_step(self) -> None:
        sampler = self.two_bucket_sampler()
        first = sampler.plan_epoch(epoch=0)
        assert first == self.two_bucket_sampler().plan_epoch(epoch=0)
        assert first != sampler.plan_epoch(epoch=1)
        assert first != self.two_bucket_sampler(seed=6).plan_epoch(epoch=0)

    def test_an_epoch_covers_every_sample_exactly_once(self) -> None:
        sampler = self.two_bucket_sampler()
        drawn = [
            index for batch in sampler.plan_epoch(epoch=0) for index in batch.indices
        ]
        assert sorted(drawn) == list(range(32))

    def test_batches_interleave_the_buckets(self) -> None:
        # Without the final shuffle the model sees all of bucket zero and then
        # all of bucket one: a resolution schedule nobody asked for.
        ids = [
            batch.bucket_id for batch in self.two_bucket_sampler().plan_epoch(epoch=0)
        ]
        transitions = sum(1 for a, b in pairwise(ids) if a != b)
        assert transitions > 1

    def test_oversampling_is_opt_out(self) -> None:
        curriculum = BucketCurriculum(
            phases=(CurriculumPhase(start_step=0, weights=(0.0, 1.0)),)
        )
        strict = self.two_bucket_sampler(
            curriculum=curriculum, allow_oversampling=False
        )
        with pytest.raises(ValueError, match="allow_oversampling"):
            strict.quotas_at(0)
        assert self.two_bucket_sampler(curriculum=curriculum).quotas_at(0) == (0, 16)

    def test_an_empty_bucket_is_masked_out_of_the_mixture(self) -> None:
        plan = BucketPlan(
            buckets=(
                Bucket(bucket_id=0, frames=4, height=4, width=4),
                Bucket(bucket_id=1, frames=8, height=4, width=4),
            )
        )
        sampler = BucketSampler(plan, [0] * 8, batch_size=2)
        assert sampler.weights_at(0) == pytest.approx((1.0, 0.0))
        assert sampler.pool_size(1) == 0

    def test_a_bucket_smaller_than_one_batch_is_refused(self) -> None:
        plan = BucketPlan(buckets=(Bucket(bucket_id=0, frames=4, height=4, width=4),))
        with pytest.raises(ValueError, match="ragged batch"):
            BucketSampler(plan, [0, 0, 0], batch_size=4)


def make_stream(length: int, *, width: int = 4, offset: float = 0.0) -> TokenStream:
    layout = PatchLayout(frames=length, height=2, width=2, patch_frames=1)
    tokens = torch.arange(length * width, dtype=torch.float32).view(1, length, width)
    return TokenStream(
        tokens=tokens + offset,
        coords=torch.full((1, length, 3), offset),
        mask=torch.ones((1, length), dtype=torch.bool),
        noise_level=torch.tensor([offset / 100.0]),
        layout=layout,
    )


class TestPacking:
    def test_the_block_diagonal_mask_matches_the_segment_ids(self) -> None:
        # Checked element by element rather than against a second construction:
        # one token of one clip attending to another clip is a data leak that
        # nothing downstream detects.
        packed, layout = pack_streams(
            [
                make_stream(3),
                make_stream(4, offset=100.0),
                make_stream(2, offset=200.0),
            ],
            capacity=12,
        )
        assert packed.tokens.shape == (1, 12, 4)
        ids = layout.segment_ids().tolist()
        assert ids == [0] * 3 + [1] * 4 + [2] * 2 + [PADDING_SEGMENT] * 3
        mask = layout.attention_mask()
        assert mask.shape == (12, 12)
        for row in range(12):
            for column in range(12):
                expected = ids[row] != PADDING_SEGMENT and ids[row] == ids[column]
                assert bool(mask[row, column]) is expected, (row, column)

    def test_cu_seqlens_bounds_every_segment(self) -> None:
        _, layout = pack_streams(
            [make_stream(3), make_stream(4), make_stream(2)], capacity=12
        )
        assert layout.cu_seqlens.tolist() == [0, 3, 7, 9]
        assert layout.cu_seqlens.dtype is torch.int32
        assert layout.offsets == (0, 3, 7)
        assert layout.segment_count == 3
        assert layout.occupancy == pytest.approx(9 / 12)

    def test_segment_weights_make_every_clip_count_equally(self) -> None:
        _, layout = pack_streams([make_stream(3), make_stream(6)], capacity=10)
        weights = layout.segment_weights()
        for start, length in zip(layout.offsets, layout.lengths, strict=True):
            assert weights[start : start + length].sum().item() == pytest.approx(1.0)
        assert weights[9].item() == 0.0
        assert layout.valid_mask().tolist() == [True] * 9 + [False]

    def test_split_packed_inverts_pack_streams(self) -> None:
        streams = [
            make_stream(3),
            make_stream(4, offset=100.0),
            make_stream(2, offset=7.0),
        ]
        packed, layout = pack_streams(streams, capacity=16)
        restored = split_packed(packed.tokens, layout)
        assert len(restored) == len(streams)
        for original, piece in zip(streams, restored, strict=True):
            assert torch.equal(piece, original.tokens)
        # The padding tail is genuinely dropped, not folded into a segment.
        assert sum(piece.shape[1] for piece in restored) == 9

    def test_packing_forces_per_token_noise(self) -> None:
        # Each segment carries its own timestep; a per-sample noise level cannot
        # express that, so the packed stream must be per-token.
        packed, layout = pack_streams(
            [make_stream(3), make_stream(4, offset=100.0)], capacity=8
        )
        assert packed.per_token_noise
        assert packed.noise_level[0, :3].tolist() == [0.0, 0.0, 0.0]
        assert packed.noise_level[0, 3:7].tolist() == pytest.approx([1.0] * 4)
        assert packed.mask[0].tolist() == [True] * 7 + [False]
        assert layout.layouts[0].frames == 3

    def test_split_packed_rejects_a_mismatched_capacity(self) -> None:
        _, layout = pack_streams([make_stream(3)], capacity=8)
        with pytest.raises(ValueError, match="capacity 8"):
            split_packed(torch.zeros(1, 7, 4), layout)

    def test_pack_streams_rejects_a_batched_or_mismatched_stream(self) -> None:
        wide = make_stream(3, width=8)
        with pytest.raises(ValueError, match="width"):
            pack_streams([make_stream(3), wide], capacity=12)
        with pytest.raises(ValueError, match="capacity"):
            pack_streams([make_stream(3), make_stream(4)], capacity=5)
        with pytest.raises(ValueError, match="at least one stream"):
            pack_streams([], capacity=8)

    def test_first_fit_decreasing_packs_tighter_than_sequential(self) -> None:
        lengths = [7, 3, 5, 5, 3, 1]
        greedy = plan_packing(lengths, capacity=8)
        ordered = plan_packing(lengths, capacity=8, strategy="sequential")
        assert greedy.occupancy >= ordered.occupancy
        for plan in (greedy, ordered):
            packed = sorted(index for bin_ in plan.bins for index in bin_)
            assert packed == list(range(len(lengths)))
            for contents in plan.bins:
                assert sum(lengths[index] for index in contents) <= 8

    def test_sequential_packing_preserves_order(self) -> None:
        plan = plan_packing([3, 3, 3, 3], capacity=6, strategy="sequential")
        assert plan.bins == ((0, 1), (2, 3))
        layout = plan.layout_for(0)
        assert layout.lengths == (3, 3)
        assert layout.sample_indices == (0, 1)

    def test_a_sample_larger_than_the_capacity_cannot_be_packed(self) -> None:
        with pytest.raises(ValueError, match="cannot be packed at all"):
            plan_packing([4, 9], capacity=8)
        with pytest.raises(ValueError, match="unknown packing strategy"):
            plan_packing([4], capacity=8, strategy="magic")
        with pytest.raises(ValueError, match="at least one sample"):
            plan_packing([], capacity=8)


class TestSyntheticSourceContract:
    def test_a_seed_pins_the_corpus(self) -> None:
        left = make_source(num_samples=4, seed=1)[2]
        right = make_source(num_samples=4, seed=1)[2]
        assert torch.equal(left.video, right.video)
        assert not torch.equal(left.video, make_source(num_samples=4, seed=2)[2].video)

    def test_the_stream_cursor_refuses_a_changed_rank_layout(self) -> None:
        source = make_source(num_samples=8)
        state = dict(source.state_dict())
        state["data_world"] = 4
        with pytest.raises(ValueError, match="data_world"):
            source.load_state_dict(state)

    def test_ranks_own_a_strided_slice_of_the_corpus(self) -> None:
        # A contiguous split would give rank zero only the low sample ids, and
        # anything that correlates with id becomes a per-rank bias.
        batches = []
        for rank in range(2):
            source = SyntheticSource(
                SyntheticConfig(**TINY, num_samples=8),
                batch_size=2,
                data_rank=rank,
                data_world=2,
            )
            batches.append([int(value) for value in next(iter(source)).sample_ids])
        assert batches == [[0, 2], [1, 3]]
