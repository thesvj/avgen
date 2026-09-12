"""Tests for the pixel/latent boundary: codecs, normalisation, and the mel frontend.

Everything below this layer knows what a pixel is and everything above it does
not, so the properties asserted here are the ones the rest of the framework is
entitled to assume without checking. Three of them carry most of the weight.

**Exact invertibility.** The reference codecs exist so a round-trip test is a
real test. A randomly-initialised stand-in would round-trip to nothing in
particular and a genuine bug in the encode/decode plumbing would sail through
green, so the assertions here are tight (``1e-5`` on float32, which is a few
ulps at these magnitudes) rather than "approximately".

**Fingerprints.** Latents from two different autoencoders have incompatible
geometry *and* incompatible statistics. Mixed into one batch they train to a
mediocre loss forever and generate nothing recognisable, with no error anywhere.
The fingerprint is the only thing standing between a shard writer and that
failure, so it must be a pure function of identity and config and it must differ
whenever anything that touches the latents differs.

**Convention fidelity.** A mel spectrogram built with the wrong scale or the
wrong normalisation feeds a vocoder trained on the other convention and produces
a plausible-but-wrong timbre rather than an obvious failure. The Slaney
assertions below pin the convention numerically.

All of it runs on CPU in a couple of seconds; the ``gpu``-marked tests only
re-check device placement, which is the one thing a CPU run cannot cover.
"""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
from typing import Any

import pytest
import torch

from avgen.codecs import (
    DiffusersVideoCodec,
    LatentStatistics,
    MelConfig,
    MelFrontend,
    ReferenceAudioCodec,
    ReferenceTextEncoder,
    ReferenceVideoCodec,
    TilingConfig,
    TransformersTextEncoder,
    blend_tiles,
    codec_fingerprint,
    denormalize_latents,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    normalize_latents,
    resolve_latent_normalization,
    stft_magnitude,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: Round-trip tolerance. The reference codecs are exactly invertible in real
#: arithmetic, so anything above a few float32 ulps at unit magnitude is a bug
#: in the geometry rather than accumulated error.
EXACT = 1e-5


def video_codec(**overrides: Any) -> ReferenceVideoCodec:
    """Return a small reference video codec, overridable per test."""
    fields: dict[str, Any] = {
        "channels": 3,
        "temporal_compression": 2,
        "spatial_compression": 2,
        "scale": 1.0,
    }
    fields.update(overrides)
    return ReferenceVideoCodec(**fields)


class TestReferenceVideoCodec:
    """The video reference codec must be exactly invertible.

    It stands in for a real VAE everywhere in CI, the simulator, and every smoke
    run. If ``decode(encode(x)) != x`` then every test that uses it is asserting
    against a moving target, so this is the load-bearing property of the whole
    module.
    """

    def test_round_trip_is_exact(self) -> None:
        codec = video_codec()
        pixels = torch.randn(2, 3, 8, 16, 16)
        restored = codec.decode(codec.encode(pixels))
        error = float((restored - pixels).abs().max())
        assert error < EXACT, f"round-trip error {error:.3e} exceeds {EXACT}"

    def test_round_trip_is_exact_under_a_non_unit_scale(self) -> None:
        # The scale divides on decode. A codec that applied it on only one side
        # would still round-trip *shapes*, so the shape assertions below cannot
        # catch it and this one has to.
        codec = video_codec(scale=1.3)
        pixels = torch.randn(2, 3, 4, 8, 8)
        error = float((codec.decode(codec.encode(pixels)) - pixels).abs().max())
        assert error < EXACT, f"round-trip error {error:.3e} exceeds {EXACT}"

    def test_encode_produces_the_shape_latent_shape_promises(self) -> None:
        # Memory estimates, patch layouts, and token counts are all computed
        # from latent_shape() without ever calling encode, so the two agreeing
        # is what makes those numbers true.
        codec = video_codec()
        pixels = torch.randn(2, 3, 8, 16, 16)
        assert tuple(codec.encode(pixels).shape) == codec.latent_shape(
            tuple(pixels.shape)
        )

    def test_latent_channels_absorb_the_whole_compression_factor(self) -> None:
        # The rearrangement is lossless, so every pixel value has to land
        # somewhere: channels widen by exactly the product of the factors.
        codec = video_codec(channels=3, temporal_compression=2, spatial_compression=2)
        assert codec.latent_channels == 3 * 2 * 2 * 2

    def test_pixel_shape_inverts_latent_shape(self) -> None:
        codec = video_codec()
        pixel_shape = (2, 3, 8, 16, 16)
        assert codec.pixel_shape(codec.latent_shape(pixel_shape)) == pixel_shape

    def test_the_channel_mixing_is_not_a_copy(self) -> None:
        # Without the orthogonal mixing, latent channel k would be a verbatim
        # copy of a pixel and a bug that transposed or dropped channels would
        # still round-trip. This asserts the mixing actually happened.
        codec = video_codec(channels=1, temporal_compression=1, spatial_compression=1)
        pixels = torch.randn(1, 1, 2, 2, 2)
        assert not torch.allclose(codec.encode(pixels), pixels)

    def test_the_mixing_preserves_the_norm(self) -> None:
        # A Householder reflection is orthogonal, which is what makes the
        # latent statistics exactly (0, scale) with nothing to measure.
        codec = video_codec(scale=1.0)
        pixels = torch.randn(2, 3, 4, 8, 8)
        latents = codec.encode(pixels)
        assert float(latents.norm()) == pytest.approx(float(pixels.norm()), rel=1e-5)

    def test_latent_statistics_report_the_scale(self) -> None:
        codec = video_codec(scale=2.5)
        statistics = codec.latent_statistics
        assert statistics.channels == codec.latent_channels
        assert statistics.std == (2.5,) * codec.latent_channels
        assert not statistics.is_identity

    @pytest.mark.parametrize(
        ("shape", "message"),
        [
            ((2, 3, 8, 16), "rank 5"),
            ((2, 4, 8, 16, 16), "pixel channels must be 3"),
            ((2, 3, 7, 16, 16), "frames=7 must be divisible"),
            ((2, 3, 8, 15, 16), "height=15 must be divisible"),
        ],
    )
    def test_rejects_a_shape_it_cannot_compress_exactly(
        self, shape: tuple[int, ...], message: str
    ) -> None:
        # Silent padding or truncation here would desynchronise audio from video
        # by a fraction of a frame, which is exactly what the exact-rational
        # timebases elsewhere in the framework exist to prevent.
        with pytest.raises(ValueError, match=message):
            video_codec().latent_shape(shape)

    def test_rejects_an_integer_tensor(self) -> None:
        with pytest.raises(TypeError, match="floating point"):
            video_codec().encode(torch.zeros(2, 3, 8, 16, 16, dtype=torch.int64))

    def test_decode_rejects_latents_from_another_codec(self) -> None:
        # Wrong latent width is the cheap, detectable half of "latents from two
        # different VAEs"; the fingerprint catches the half that has the right
        # width and the wrong meaning.
        with pytest.raises(ValueError, match="latent channels must be"):
            video_codec().decode(torch.randn(2, 7, 4, 8, 8))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"channels": 0},
            {"temporal_compression": -1},
            {"spatial_compression": 0},
            {"scale": 0.0},
            {"scale": math.inf},
        ],
    )
    def test_rejects_an_invalid_configuration(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            video_codec(**kwargs)


class TestReferenceAudioCodec:
    """The audio reference codec must be exactly invertible too.

    Non-overlapping framing rather than an STFT is what makes that possible: an
    STFT with a non-rectangular window is only invertible up to edge effects and
    a COLA condition, which would make this a weaker assertion than it is.
    """

    def test_round_trip_is_exact(self) -> None:
        codec = ReferenceAudioCodec(channels=1, hop_length=16, sample_rate=16000)
        waveform = torch.randn(2, 1, 512)
        error = float((codec.decode(codec.encode(waveform)) - waveform).abs().max())
        assert error < EXACT, f"round-trip error {error:.3e} exceeds {EXACT}"

    def test_round_trip_is_exact_for_multi_channel_audio(self) -> None:
        codec = ReferenceAudioCodec(channels=2, hop_length=8, scale=0.75)
        waveform = torch.randn(3, 2, 64)
        error = float((codec.decode(codec.encode(waveform)) - waveform).abs().max())
        assert error < EXACT, f"round-trip error {error:.3e} exceeds {EXACT}"

    def test_latent_geometry_is_self_consistent(self) -> None:
        codec = ReferenceAudioCodec(channels=1, hop_length=16, sample_rate=16000)
        assert codec.latent_channels == 16
        assert codec.latent_frames(512) == 32
        assert codec.num_samples(32) == 512
        assert codec.latent_rate == pytest.approx(1000.0)

    def test_rejects_a_partial_hop(self) -> None:
        # Truncating here would drift audio against video by a fraction of a
        # frame per clip, and nothing downstream would report it.
        codec = ReferenceAudioCodec(hop_length=16)
        with pytest.raises(ValueError, match="must be divisible by hop_length"):
            codec.latent_frames(500)

    def test_rejects_the_wrong_rank(self) -> None:
        codec = ReferenceAudioCodec(hop_length=16)
        with pytest.raises(ValueError, match="rank 3"):
            codec.encode(torch.randn(2, 512))

    def test_rejects_the_wrong_channel_count(self) -> None:
        codec = ReferenceAudioCodec(channels=1, hop_length=16)
        with pytest.raises(ValueError, match="waveform channels must be 1"):
            codec.encode(torch.randn(2, 2, 512))


class TestReferenceTextEncoder:
    """The stand-in text encoder must be deterministic and order-sensitive.

    A hash of the whole prompt broadcast across the token axis would be simpler,
    but it makes every token of a prompt identical, so a cross-attention bug
    that collapsed the text axis would still pass every test written against it.
    """

    def test_is_deterministic(self) -> None:
        encoder = ReferenceTextEncoder(width=32, max_length=12)
        first, first_mask = encoder.encode(["a cat", "a dog"])
        second, second_mask = encoder.encode(["a cat", "a dog"])
        assert torch.equal(first, second)
        assert torch.equal(first_mask, second_mask)

    def test_an_empty_prompt_is_the_null_context(self) -> None:
        # An all-empty batch must be bit-identical to the classifier-free
        # guidance null branch, or "unconditional" means two different things in
        # two different code paths.
        encoder = ReferenceTextEncoder(width=32, max_length=12)
        features, mask = encoder.encode([""])
        assert not bool(mask.any())
        assert float(features.abs().max()) == 0.0

    def test_byte_order_changes_the_features(self) -> None:
        encoder = ReferenceTextEncoder(width=32, max_length=12)
        features, _ = encoder.encode(["ab", "ba"])
        assert not torch.allclose(features[0], features[1])

    def test_padding_positions_are_masked_and_zero(self) -> None:
        encoder = ReferenceTextEncoder(width=32, max_length=12)
        features, mask = encoder.encode(["ab"])
        assert mask[0].tolist() == [True, True] + [False] * 10
        assert float(features[0, 2:].abs().max()) == 0.0

    def test_truncates_at_the_byte_budget(self) -> None:
        encoder = ReferenceTextEncoder(width=16, max_length=4)
        _, mask = encoder.encode(["a very long prompt"])
        assert bool(mask.all())

    def test_rejects_an_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            ReferenceTextEncoder().encode([])

    def test_rejects_a_non_string_prompt(self) -> None:
        with pytest.raises(TypeError, match=r"prompts\[1\] must be a string"):
            ReferenceTextEncoder().encode(["ok", 3])  # type: ignore[list-item]


class TestCodecFingerprint:
    """Fingerprints are what stop latents from two codecs entering one batch.

    The failure they prevent is quiet: incompatible geometry and statistics mixed
    into one batch produce a model that trains to a mediocre loss forever and
    generates nothing recognisable, with no error raised anywhere. That means the
    fingerprint has to be a pure function of identity and config — never of the
    process, the device, or the wall clock — because it is compared across
    machines and across months.
    """

    def test_is_stable_across_instances(self) -> None:
        first = video_codec(scale=1.3)
        second = video_codec(scale=1.3)
        assert first.fingerprint == second.fingerprint

    def test_is_stable_across_calls_with_reordered_config_keys(self) -> None:
        # The digest is taken over canonical JSON with sorted keys, so two
        # processes building the same codec from the same arguments agree byte
        # for byte regardless of mapping iteration order.
        assert codec_fingerprint("video", "x", {"a": 1, "b": 2}) == codec_fingerprint(
            "video", "x", {"b": 2, "a": 1}
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"channels": 1},
            {"temporal_compression": 4},
            {"spatial_compression": 4},
            {"scale": 1.3},
            {"seed": 7},
        ],
    )
    def test_differs_whenever_anything_touching_the_latents_differs(
        self, overrides: dict[str, Any]
    ) -> None:
        assert video_codec().fingerprint != video_codec(**overrides).fingerprint

    def test_carries_the_kind_and_identity_in_the_clear(self) -> None:
        # A shard header and a log line both have to be readable by a human
        # debugging a mismatch, so the prefix is not hashed.
        fingerprint = video_codec().fingerprint
        kind, identity, digest = fingerprint.split(":")
        assert (kind, identity) == ("video", "avgen.reference")
        assert len(digest) == 16

    def test_two_codecs_of_different_kinds_never_collide(self) -> None:
        audio = ReferenceAudioCodec()
        text = ReferenceTextEncoder()
        assert (
            len({video_codec().fingerprint, audio.fingerprint, text.fingerprint}) == 3
        )

    @pytest.mark.parametrize("bad", ["", " video", "video ", "\t"])
    def test_rejects_an_id_that_media_batch_spec_would_reject(self, bad: str) -> None:
        with pytest.raises(ValueError, match="non-empty trimmed string"):
            codec_fingerprint(bad, "identity")

    def test_latents_from_a_different_codec_cannot_be_normalised(self) -> None:
        # The fingerprint catches the mixing at the batch assembler; this is the
        # second line of defence, at the point the statistics are applied.
        statistics = video_codec(scale=1.3).latent_statistics
        foreign = torch.randn(2, 5, 4, 8, 8)
        with pytest.raises(ValueError, match="must never be normalised"):
            normalize_latents(foreign, statistics)

    def test_identity_statistics_still_check_the_width(self) -> None:
        # The check runs before the identity fast path. Identity statistics are
        # numerically a no-op at any width, so skipping the check is harmless
        # arithmetically — but it removes this second line of defence for every
        # codec whose latent space is already standardised, which includes both
        # reference codecs. A width mismatch means the wrong codec, not a
        # harmless shape.
        statistics = video_codec(scale=1.0).latent_statistics
        assert statistics.is_identity
        foreign = torch.randn(2, 5, 4, 8, 8)
        with pytest.raises(ValueError, match="must never be normalised"):
            normalize_latents(foreign, statistics)
        with pytest.raises(ValueError, match="must never be normalised"):
            denormalize_latents(foreign, statistics)

    def test_identity_statistics_still_return_the_same_tensor(self) -> None:
        # Passing the check must not cost a copy: a bf16 latent cache has to
        # stay bit-identical rather than round-tripping through a no-op multiply.
        statistics = video_codec(scale=1.0).latent_statistics
        latents = torch.randn(2, statistics.channels, 4, 8, 8, dtype=torch.bfloat16)
        assert normalize_latents(latents, statistics) is latents
        assert denormalize_latents(latents, statistics) is latents


class TestLatentNormalization:
    """Normalisation must be exactly invertible and applied in exactly one place.

    Forgetting to denormalise before decoding hands the decoder latents from a
    distribution it has never seen, which is the single most common cause of
    "the model trains fine but the decoded video is grey mush". The inverse
    holding exactly is what makes that a code-path question rather than a
    numerical one.
    """

    def test_round_trip_returns_the_original(self) -> None:
        statistics = LatentStatistics(mean=(0.5, -1.0, 2.0), std=(2.0, 0.5, 3.0))
        latents = torch.randn(2, 3, 4, 5)
        restored = denormalize_latents(
            normalize_latents(latents, statistics), statistics
        )
        torch.testing.assert_close(restored, latents, rtol=0, atol=1e-5)

    def test_normalising_actually_standardises(self) -> None:
        # The point of the operation: flow matching interpolates against unit
        # noise, so a latent space with std 6 is never shown the high-noise
        # regime the sampler starts from.
        latents = torch.randn(4, 3, 8, 8) * 6.0 + 2.0
        statistics = LatentStatistics.from_latents(latents)
        normalised = normalize_latents(latents, statistics)
        # Population std, matching from_latents: the difference from the sample
        # std is 1/2N, which at these sizes is larger than the tolerance worth
        # asserting at.
        per_channel_std = (
            normalised.transpose(0, 1).reshape(3, -1).std(dim=1, unbiased=False)
        )
        torch.testing.assert_close(per_channel_std, torch.ones(3), rtol=0, atol=1e-5)

    def test_identity_statistics_are_a_no_op_by_identity_not_by_value(self) -> None:
        # Skipping is not just an optimisation: it keeps a bf16 latent cache
        # bit-identical rather than round-tripping through a no-op multiply.
        statistics = LatentStatistics.identity(3)
        latents = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
        assert normalize_latents(latents, statistics) is latents
        assert denormalize_latents(latents, statistics) is latents

    def test_preserves_the_storage_dtype(self) -> None:
        statistics = LatentStatistics.from_scalar(3, mean=1.0, std=2.0)
        latents = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
        assert normalize_latents(latents, statistics).dtype is torch.bfloat16

    def test_from_scalar_is_the_diffusers_convention_per_channel(self) -> None:
        statistics = LatentStatistics.from_scalar(4, mean=0.5, std=0.25)
        assert statistics.mean == (0.5,) * 4
        assert statistics.std == (0.25,) * 4

    def test_survives_a_serialisation_round_trip(self) -> None:
        # These land in a checkpoint manifest and a pinned generation config, so
        # the JSON form has to reconstruct exactly.
        statistics = LatentStatistics(mean=(0.5, -1.0), std=(2.0, 0.5), channel_dim=1)
        assert LatentStatistics.from_dict(statistics.to_dict()) == statistics

    def test_tensors_broadcast_against_the_channel_axis(self) -> None:
        statistics = LatentStatistics(mean=(0.0, 1.0), std=(1.0, 2.0), channel_dim=1)
        mean, std = statistics.tensors(ndim=5)
        assert tuple(mean.shape) == (1, 2, 1, 1, 1)
        assert tuple(std.shape) == (1, 2, 1, 1, 1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mean": (0.0,), "std": (1.0, 2.0)},
            {"mean": (), "std": ()},
            {"mean": (0.0,), "std": (0.0,)},
            {"mean": (0.0,), "std": (-1.0,)},
            {"mean": (math.nan,), "std": (1.0,)},
            {"mean": (0.0,), "std": (1.0,), "channel_dim": -1},
        ],
    )
    def test_rejects_statistics_that_cannot_be_applied(
        self, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            LatentStatistics(**kwargs)

    def test_rejects_a_channel_axis_the_tensor_does_not_have(self) -> None:
        statistics = LatentStatistics(mean=(0.0, 1.0), std=(1.0, 2.0), channel_dim=4)
        with pytest.raises(ValueError, match="not a valid axis"):
            normalize_latents(torch.randn(2, 2, 3), statistics)


class TestResolveLatentNormalization:
    """The fallback order must be identical everywhere it is used.

    A silent fallback to identity in the inference path and to a scaling factor
    in the training path is a train/inference skew bug whose only symptom is
    slightly washed-out samples, so the ordering is pinned here step by step.
    """

    def test_1_latent_statistics_pass_through_unchanged(self) -> None:
        statistics = LatentStatistics(mean=(0.0, 1.0), std=(1.0, 2.0))
        assert resolve_latent_normalization(statistics) is statistics

    def test_2_a_codec_is_recursed_into(self) -> None:
        codec = video_codec(scale=1.3)
        assert resolve_latent_normalization(codec) == codec.latent_statistics

    def test_3_a_mapping_with_mean_and_std_is_per_channel(self) -> None:
        resolved = resolve_latent_normalization({"mean": [0.0, 1.0], "std": [1.0, 2.0]})
        assert resolved == LatentStatistics(mean=(0.0, 1.0), std=(1.0, 2.0))

    def test_4_a_scaling_factor_is_inverted_into_a_standard_deviation(self) -> None:
        # diffusers *multiplies* by scaling_factor on encode, so it plays the
        # role of 1 / std. Getting the direction wrong scales every latent by
        # the square of the factor and is invisible until the decode.
        resolved = resolve_latent_normalization(
            {"scaling_factor": 0.5, "shift_factor": 0.25}, channels=2
        )
        assert resolved == LatentStatistics(mean=(0.25, 0.25), std=(2.0, 2.0))

    def test_5_a_raw_diffusers_module_config_is_read(self) -> None:
        module = _fake_vae()
        resolved = resolve_latent_normalization(module)
        assert resolved.std == (1.0 / 1.5,) * 4

    def test_6_none_is_identity_and_needs_a_width(self) -> None:
        assert resolve_latent_normalization(None, channels=3).is_identity
        with pytest.raises(ValueError, match="channels is required"):
            resolve_latent_normalization(None)

    def test_rejects_a_source_it_does_not_understand(self) -> None:
        with pytest.raises(TypeError, match="cannot resolve latent statistics"):
            resolve_latent_normalization(object())

    def test_rejects_a_mapping_with_neither_convention(self) -> None:
        with pytest.raises(TypeError, match="mean and std, or"):
            resolve_latent_normalization({"latent_channels": 4})


class TestMelFrontend:
    """The mel convention has to be pinned numerically, not described in prose.

    A vocoder trained against Slaney mels fed an HTK spectrogram produces a
    plausible but wrong timbre rather than an obvious failure, so "we implement
    Slaney" is only true if something checks the break point and the area
    normalisation.
    """

    def test_the_slaney_break_point_is_exactly_mel_15(self) -> None:
        # 200/3 Hz per mel below 1 kHz is the constant Slaney chose precisely so
        # that 1 kHz lands on mel 15 and the linear and log regions join without
        # a discontinuity. HTK would put 1 kHz near mel 1000.
        assert float(hz_to_mel(torch.tensor(1000.0))) == pytest.approx(15.0)

    def test_the_mel_scale_is_invertible(self) -> None:
        hertz = torch.tensor([0.0, 100.0, 999.0, 1000.0, 4000.0, 8000.0])
        torch.testing.assert_close(
            mel_to_hz(hz_to_mel(hertz)), hertz, rtol=1e-5, atol=1e-3
        )

    def test_the_linear_region_really_is_linear(self) -> None:
        below = torch.tensor([100.0, 200.0, 400.0])
        torch.testing.assert_close(
            hz_to_mel(below), below * 3.0 / 200.0, rtol=1e-6, atol=0
        )

    def test_zero_hertz_does_not_produce_a_nan(self) -> None:
        # torch.where evaluates both branches, so the log argument has to be
        # clamped even where it is discarded.
        assert bool(torch.isfinite(hz_to_mel(torch.zeros(1))).all())

    def test_filterbank_shape(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        assert tuple(mel_filterbank(config).shape) == (20, config.num_bins)

    def test_every_band_has_support_and_no_negative_weights(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=512, hop_length=128, n_mels=24)
        bank = mel_filterbank(config)
        assert bool((bank >= 0).all())
        assert int((bank.sum(dim=1) > 0).sum()) == 24, "a band with no support is dead"

    def test_slaney_area_normalisation_rather_than_unit_peak(self) -> None:
        # Each band is scaled to unit *area*, so its peak is 2 / (f[m+2] - f[m])
        # and a wide high-frequency band does not dominate a narrow low one
        # purely by covering more bins. Skipping this normalisation is the usual
        # reason a home-rolled mel does not match librosa, and it would leave
        # every peak at exactly 1.0 instead.
        config = MelConfig(sample_rate=16000, n_fft=512, hop_length=128, n_mels=24)
        bank = mel_filterbank(config)
        edges_mel = torch.linspace(
            float(hz_to_mel(torch.tensor(config.f_min, dtype=torch.float64))),
            float(hz_to_mel(torch.tensor(config.upper_hz, dtype=torch.float64))),
            config.n_mels + 2,
            dtype=torch.float64,
        )
        edges_hz = mel_to_hz(edges_mel)
        ceiling = (2.0 / (edges_hz[2:] - edges_hz[:-2])).to(torch.float32)
        peaks = bank.max(dim=1).values
        # The discrete bin grid rarely samples a triangle's apex exactly, so the
        # realised peak sits just below the analytic one rather than on it.
        assert bool((peaks <= ceiling * (1.0 + 1e-5)).all())
        assert bool((peaks >= ceiling * 0.5).all())
        assert float(peaks.max()) < 0.5, "peaks near 1.0 would mean HTK unit-peak"

    def test_stft_is_deterministic(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        waveform = torch.randn(2, 1, 512)
        assert torch.equal(
            stft_magnitude(waveform, config), stft_magnitude(waveform, config)
        )

    def test_stft_frame_count_matches_frames_for(self) -> None:
        # Everything downstream sizes an audio latent grid from frames_for()
        # without running the transform, so a disagreement here is a shape bug
        # that only surfaces at the first real batch.
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        magnitude = stft_magnitude(torch.randn(2, 512), config)
        assert tuple(magnitude.shape) == (2, config.num_bins, config.frames_for(512))

    @pytest.mark.parametrize("shape", [(512,), (2, 512), (2, 1, 512)])
    def test_accepts_every_documented_input_rank(self, shape: tuple[int, ...]) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        magnitude = stft_magnitude(torch.randn(shape), config)
        assert magnitude.ndim == 3

    def test_refuses_to_silently_downmix_stereo(self) -> None:
        # Averaging channels here would hide a stereo file reaching a mono
        # pipeline, which is a data bug that should be loud.
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64)
        with pytest.raises(ValueError, match="expects one channel"):
            stft_magnitude(torch.randn(2, 2, 512), config)

    def test_frontend_produces_the_declared_band_count(self) -> None:
        frontend = MelFrontend(
            MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        )
        spectrogram = frontend(torch.randn(2, 1, 512))
        assert tuple(spectrogram.shape) == (2, 20, frontend.config.frames_for(512))
        assert bool((spectrogram >= 0).all())

    def test_log_mel_floors_silence_instead_of_producing_minus_infinity(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        frontend = MelFrontend(config)
        silence = frontend.log_mel(torch.zeros(1, 1, 512))
        assert bool(torch.isfinite(silence).all())
        assert float(silence.max()) == pytest.approx(math.log(config.log_epsilon))

    def test_the_filterbank_cache_returns_an_equal_matrix(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        assert torch.equal(mel_filterbank(config), mel_filterbank(config))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_fft": 0},
            {"hop_length": 0},
            {"n_mels": 0},
            {"win_length": 4096},
            {"power": 0.0},
            {"log_epsilon": 0.0},
            {"f_min": -1.0},
            {"f_max": 1e9},
            {"f_min": 4000.0, "f_max": 100.0},
        ],
    )
    def test_rejects_an_unusable_configuration(self, kwargs: dict[str, Any]) -> None:
        fields: dict[str, Any] = {"sample_rate": 16000, "n_fft": 256, "hop_length": 64}
        fields.update(kwargs)
        with pytest.raises(ValueError):
            MelConfig(**fields)

    def test_frames_for_rejects_a_clip_shorter_than_one_window(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, center=False)
        with pytest.raises(ValueError, match="leaves no complete frame"):
            config.frames_for(100)


class _FakeVAEConfig:
    """The subset of a diffusers VAE config the adapter actually reads."""

    latent_channels = 4
    scaling_factor = 1.5
    shift_factor = 0.0
    out_channels = 1
    block_out_channels = (1, 2)
    spatial_compression_ratio = 2
    temporal_compression_ratio = 1


class _Posterior:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latents = latents

    def mode(self) -> torch.Tensor:
        return self.latents


class _EncodeResult:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latent_dist = _Posterior(latents)


class _DecodeResult:
    def __init__(self, pixels: torch.Tensor) -> None:
        self.sample = pixels


class _FakeVAE:
    """A pointwise stand-in for a diffusers VAE, used to test tiling.

    Deliberately *local*: a 2x2 pixel-unshuffle and a scalar gain, so the value
    a tile produces for a position does not depend on which tile it was in. A
    real convolutional decoder is not local — its receptive field spans the tile
    boundary — which is why tiling produces seams and why an exact tiled-equals-
    untiled assertion against one would be testing the wrong thing. Here it is
    testing exactly the right thing: the window offsets, the blending weights,
    and that the weights form a partition of unity.
    """

    dtype = torch.float32
    config = _FakeVAEConfig()

    def encode(self, pixels: torch.Tensor) -> _EncodeResult:
        batch, channels, frames, height, width = pixels.shape
        folded = (
            pixels.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
            .permute(0, 1, 4, 6, 2, 3, 5)
            .reshape(batch, 4 * channels, frames, height // 2, width // 2)
        )
        return _EncodeResult(folded * 1.5)

    def decode(self, latents: torch.Tensor) -> _DecodeResult:
        batch, channels, frames, height, width = latents.shape
        unfolded = (
            (latents / 1.5)
            .reshape(batch, channels // 4, 2, 2, frames, height, width)
            .permute(0, 1, 4, 5, 2, 6, 3)
            .reshape(batch, channels // 4, frames, height * 2, width * 2)
        )
        return _DecodeResult(unfolded)


def _fake_vae() -> _FakeVAE:
    """Return the pointwise stand-in VAE."""
    return _FakeVAE()


def _codec_with_fake_weights(tiling: TilingConfig) -> DiffusersVideoCodec:
    """Return a diffusers adapter with the fake module already resolved.

    Assigning ``_module`` is what a completed lazy load leaves behind, so the
    adapter runs its real encode/decode/tiling code without a download.
    """
    codec = DiffusersVideoCodec("fake/vae", tiling=tiling)
    codec._module = _fake_vae()
    return codec


class TestTiling:
    """Tiled decoding must agree with untiled decoding.

    That is the entire justification for the feature: it trades time for memory
    on a decoder that would otherwise OOM, and it is only an acceptable trade if
    the result is the same picture. The blending weights have to form a
    partition of unity for that to hold — accumulating ``output * weight`` and
    ``weight`` separately and dividing at the end is the only formulation where
    it does, including for the pulled-back final window whose overlap with its
    predecessor is not the nominal one.
    """

    def test_windows_cover_the_axis_without_a_short_final_window(self) -> None:
        # A short final tile would get a blending ramp of the wrong length and
        # leave a visible bright or dark band at the frame edge.
        windows = blend_tiles(16, 8, 4)
        assert windows[0][0] == 0
        assert windows[-1][1] == 16
        assert all(stop - start == 8 for start, stop in windows)

    def test_a_short_axis_is_one_window(self) -> None:
        assert blend_tiles(5, 8, 4) == [(0, 5)]

    def test_consecutive_windows_advance_by_the_stride(self) -> None:
        # Every window but the pulled-back last one advances by tile - overlap;
        # the last starts wherever it must to end exactly at the axis extent, so
        # its stride is smaller and its overlap correspondingly larger, which is
        # why the blend has to normalise rather than assume a fixed overlap.
        windows = blend_tiles(20, 8, 2)
        starts = [start for start, _ in windows]
        assert starts[:-1] == list(range(0, starts[-2] + 1, 6))
        assert 0 < starts[-1] - starts[-2] <= 6

    @pytest.mark.parametrize(("tile", "overlap"), [(0, 0), (4, 4), (4, 8)])
    def test_rejects_a_window_that_never_advances(
        self, tile: int, overlap: int
    ) -> None:
        with pytest.raises(ValueError):
            blend_tiles(32, tile, overlap)

    def test_tiled_encode_matches_untiled(self) -> None:
        pixels = torch.randn(1, 1, 4, 16, 16)
        plain = _codec_with_fake_weights(TilingConfig())
        tiled = _codec_with_fake_weights(
            TilingConfig(enabled=True, tile_size=8, tile_overlap=4)
        )
        torch.testing.assert_close(
            tiled.encode(pixels), plain.encode(pixels), rtol=0, atol=EXACT
        )

    def test_tiled_decode_matches_untiled(self) -> None:
        plain = _codec_with_fake_weights(TilingConfig())
        latents = plain.encode(torch.randn(1, 1, 4, 16, 16))
        tiled = _codec_with_fake_weights(
            TilingConfig(enabled=True, tile_size=8, tile_overlap=4)
        )
        torch.testing.assert_close(
            tiled.decode(latents), plain.decode(latents), rtol=0, atol=EXACT
        )

    def test_temporal_chunking_matches_unchunked(self) -> None:
        # A temporal chunk boundary produces a visible flicker at the join; the
        # same overlap-and-blend fix has to be exact for the same reason the
        # spatial one does.
        pixels = torch.randn(1, 1, 8, 16, 16)
        plain = _codec_with_fake_weights(TilingConfig())
        tiled = _codec_with_fake_weights(
            TilingConfig(
                enabled=True,
                tile_size=8,
                tile_overlap=4,
                temporal_chunk=4,
                temporal_overlap=2,
            )
        )
        torch.testing.assert_close(
            tiled.encode(pixels), plain.encode(pixels), rtol=0, atol=EXACT
        )

    def test_tiling_preserves_the_output_shape(self) -> None:
        pixels = torch.randn(1, 1, 4, 16, 16)
        tiled = _codec_with_fake_weights(
            TilingConfig(enabled=True, tile_size=8, tile_overlap=4)
        )
        assert tuple(tiled.encode(pixels).shape) == tiled.latent_shape(
            tuple(pixels.shape)
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tile_size": 8, "tile_overlap": 8},
            {"tile_size": 8, "tile_overlap": 16},
            {"temporal_chunk": 4, "temporal_overlap": 4},
            {"tile_size": -1},
            {"enabled": True, "tile_size": 0},
        ],
    )
    def test_rejects_an_unusable_geometry(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            TilingConfig(**kwargs)

    def test_the_geometry_is_part_of_the_fingerprint(self) -> None:
        # Tiled and untiled latents differ for a real convolutional VAE, so a
        # shard written with one must never be read as the other.
        plain = DiffusersVideoCodec("fake/vae")
        tiled = DiffusersVideoCodec("fake/vae", tiling=TilingConfig(enabled=True))
        assert plain.fingerprint != tiled.fingerprint

    def test_the_fingerprint_is_available_before_the_weights_are(self) -> None:
        # A shard writer and a training node must agree on the id without both
        # paying to instantiate the VAE.
        assert (
            DiffusersVideoCodec("fake/vae").fingerprint
            == DiffusersVideoCodec("fake/vae").fingerprint
        )

    def test_the_adapter_reads_the_geometry_off_the_loaded_config(self) -> None:
        codec = _codec_with_fake_weights(TilingConfig())
        assert codec.latent_channels == 4
        assert codec.spatial_compression == 2
        assert codec.temporal_compression == 1
        assert codec.latent_statistics.std == (1.0 / 1.5,) * 4


class TestOptionalDependencies:
    """A missing extra must name the extra, not surface as an ImportError.

    The whole point of importing ``diffusers`` and ``transformers`` inside the
    function that needs them is that a cluster image which never runs text
    encoding does not carry a 2GB dependency tree. That trade is only worth
    making if the failure mode is a message a user can act on.
    """

    @pytest.fixture
    def no_optional_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make ``diffusers`` and ``transformers`` unimportable.

        Patched rather than relying on them being absent, so the test asserts
        the same thing on a machine that happens to have the extras installed —
        where the unpatched call would otherwise try to reach the Hub.
        """
        real = importlib.import_module

        def fake(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in {"diffusers", "transformers"}:
                raise ImportError(f"No module named {name!r}")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake)

    @pytest.mark.usefixtures("no_optional_imports")
    def test_the_video_codec_names_the_codecs_extra(self) -> None:
        with pytest.raises(RuntimeError, match=r"avgen\[codecs\]"):
            DiffusersVideoCodec("stabilityai/sd-vae").module()

    @pytest.mark.usefixtures("no_optional_imports")
    def test_a_property_that_needs_the_weights_also_raises_runtime_error(self) -> None:
        # latent_channels reaches the module lazily, so a caller that never
        # touches module() directly must still get the actionable message.
        with pytest.raises(RuntimeError, match=r"avgen\[codecs\]"):
            _ = DiffusersVideoCodec("stabilityai/sd-vae").latent_channels

    @pytest.mark.usefixtures("no_optional_imports")
    def test_the_text_tokenizer_names_the_text_extra(self) -> None:
        with pytest.raises(RuntimeError, match=r"avgen\[text\]"):
            TransformersTextEncoder("google/t5-v1_1-xxl").tokenizer()

    @pytest.mark.usefixtures("no_optional_imports")
    def test_the_text_model_names_the_text_extra(self) -> None:
        with pytest.raises(RuntimeError, match=r"avgen\[text\]"):
            TransformersTextEncoder("google/t5-v1_1-xxl").model()

    @pytest.mark.slow
    def test_importing_the_package_needs_nothing_but_torch(self) -> None:
        """Importing avgen.codecs must not import an optional dependency.

        The steady-state training hot path must never touch one, and the
        cheapest way to keep that true is that the import does not either.

        Checked in a **fresh interpreter**, not against this process's
        ``sys.modules``. In-process the assertion is a statement about whichever
        tests happened to run first — any earlier test that legitimately imports
        transformers makes it fail — so it would pass or fail on test order and
        xdist sharding rather than on the property it names.
        """
        probe = (
            "import sys; import avgen.codecs; "
            "leaked = [n for n in ('diffusers', 'transformers') if n in sys.modules]; "
            "print(','.join(leaked))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip() == "", (
            f"importing avgen.codecs pulled in {result.stdout.strip()}"
        )

    @pytest.mark.parametrize("model_id", ["", " x", "x "])
    def test_an_untrimmed_model_id_is_rejected_at_construction(
        self, model_id: str
    ) -> None:
        # Construction is offline, so this is the one chance to reject a typo
        # before it becomes a 404 at the top of a training run.
        with pytest.raises(ValueError, match="non-empty trimmed string"):
            DiffusersVideoCodec(model_id)

    def test_a_non_positive_text_budget_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_length must be"):
            TransformersTextEncoder("google/t5-v1_1-xxl", max_length=0)


@pytest.mark.gpu
@CUDA
class TestOnDevice:
    """Device placement, which a CPU run cannot cover.

    Everything else in this file is device-independent arithmetic; these check
    that the pieces which allocate — the mixing matrix, the filterbank, the
    statistics tensors — land on the tensor's device rather than dragging it
    back to the host.
    """

    def test_video_round_trip_is_exact_on_cuda(self) -> None:
        codec = video_codec(scale=1.3)
        pixels = torch.randn(2, 3, 4, 8, 8, device="cuda")
        restored = codec.decode(codec.encode(pixels))
        assert restored.device.type == "cuda"
        error = float((restored - pixels).abs().max())
        assert error < EXACT, f"round-trip error {error:.3e} exceeds {EXACT}"

    def test_normalisation_stays_on_the_latents_device(self) -> None:
        statistics = LatentStatistics(mean=(0.5, -1.0), std=(2.0, 0.5))
        latents = torch.randn(2, 2, 4, 4, device="cuda")
        normalised = normalize_latents(latents, statistics)
        assert normalised.device == latents.device
        torch.testing.assert_close(
            denormalize_latents(normalised, statistics), latents, rtol=0, atol=1e-5
        )

    def test_the_filterbank_materialises_on_the_requested_device(self) -> None:
        config = MelConfig(sample_rate=16000, n_fft=256, hop_length=64, n_mels=20)
        frontend = MelFrontend(config)
        waveform = torch.randn(2, 1, 512, device="cuda")
        spectrogram = frontend(waveform)
        assert spectrogram.device.type == "cuda"
        torch.testing.assert_close(
            spectrogram.cpu(), MelFrontend(config)(waveform.cpu()), rtol=1e-4, atol=1e-5
        )
