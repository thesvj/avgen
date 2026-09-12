"""Tests for :mod:`avgen.finetune` — adapters, freezing, and stage recipes.

The package makes three claims in its own docstring, and this file is what makes
them cost something to break.

**Identity at initialisation is the load-bearing one.** Every adapter here is
built so that the adapted model's output is *bit-identical* to the base model's
before the first optimizer step. That is not an aesthetic property. A pretrained
video model is a balanced fixed point; a random perturbation of every projection
at step 0 produces a loss spike, and the first hundred steps go on undoing damage
instead of learning. Asserting ``torch.equal`` rather than ``allclose`` is
deliberate — an approximate identity would hide a real perturbation behind a
tolerance, and there is no numerical reason for the outputs to differ at all when
``B = 0``.

**State-dict paths must survive injection.** ``LoRALinear`` subclasses
``nn.Linear`` instead of wrapping one so that ``blocks.*.attention.q_proj.weight``
still names the same tensor after ``apply_lora``. A wrapper would rename every
base key, which at this scale means a conversion pass over a multi-terabyte
distributed checkpoint and a tensor-parallel plan that no longer addresses
anything. :class:`TestStateDictPaths` pins that down against a real checkpoint's
key set.

Everything runs on CPU against the real :class:`~avgen.models.VideoDiT` at the
``"tiny"`` preset, not a stub. A stub would let the adapters agree with a model
whose module names and call conventions nobody else uses.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn

from avgen.core import GridPatchifier, ModelInput, TextContext, TokenStream
from avgen.finetune import (
    ControlAdapter,
    ControlAdapterConfig,
    DistillationConfig,
    FinetuneStage,
    IPAdapter,
    IPAdapterConfig,
    LoRAConfig,
    LoRALinear,
    TimestepPlan,
    adapter_state_dict,
    apply_lora,
    apply_stage,
    freeze_all,
    freeze_except,
    freeze_matching,
    get_stage,
    list_stages,
    load_adapter,
    lora_modules,
    lora_parameters,
    mark_only_lora_trainable,
    merge_lora,
    register_stage,
    resolution_shift,
    save_adapter,
    shift_timesteps,
    stage_names_for,
    standard_stages,
    trainable_summary,
    unmerge_lora,
)
from avgen.models import VideoDiT, preset

# Two blocks, width 64, 4x4 latents: a full forward is milliseconds, and every
# adapter target (q/k/v/out_proj, gate/up/down_proj) is present exactly as the
# model contract names them.
FRAMES = 4
EXTENT = 4
TEXT_TOKENS = 6


def build_model(seed: int = 0) -> VideoDiT:
    """Return a tiny VideoDiT whose output is not identically zero.

    ``init_weights`` zero-initialises ``final_proj`` on purpose, so a freshly
    built model predicts exactly zero. Every identity assertion in this file
    would then pass vacuously — zero equals zero however badly an adapter
    misbehaves — so the output head is given real weights afterwards.
    """
    torch.manual_seed(seed)
    model = VideoDiT(preset("tiny"))
    model.init_weights()
    with torch.no_grad():
        nn.init.normal_(model.final_proj.weight, std=0.02)
    model.eval()
    return model


def make_inputs(model: VideoDiT, *, batch: int = 2, seed: int = 1) -> ModelInput:
    """Build one valid ModelInput for the tiny model."""
    config = model.config
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(
        (batch, config.in_channels, FRAMES, EXTENT, EXTENT), generator=generator
    )
    positions = torch.arange(FRAMES, dtype=torch.float32).repeat(batch, 1) / 8.0
    mask = torch.ones((batch, FRAMES, EXTENT, EXTENT), dtype=torch.bool)
    stream = model.patchifier.to_tokens(
        latents,
        positions=positions,
        mask=mask,
        noise_level=torch.rand(batch, generator=generator),
    )
    text = TextContext(
        features=torch.randn(
            (batch, TEXT_TOKENS, config.text_width), generator=generator
        ),
        mask=torch.ones((batch, TEXT_TOKENS), dtype=torch.bool),
    )
    inputs = ModelInput(
        video=stream,
        audio=TokenStream.empty_like(batch, stream.width),
        text=text,
    )
    inputs.validate()
    return inputs


def predict(model: VideoDiT, inputs: ModelInput) -> torch.Tensor:
    """Run the model under ``no_grad`` and return a detached copy of the output."""
    with torch.no_grad():
        return model(inputs).video.clone()


def randomise_adapters(model: nn.Module, *, seed: int = 5, std: float = 0.05) -> None:
    """Move every adapter parameter off its initialisation.

    An adapter at initialisation is the identity, so merge and round-trip tests
    against it would pass for a model with no adapter at all.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in lora_parameters(model):
            parameter.add_(
                torch.randn(parameter.shape, generator=generator) * std,
            )


class TestLoRAIdentityAtInit:
    """``apply_lora`` must not move the model at all before the first step.

    ``lora_b`` is zero-initialised, so ``B·A`` is exactly the zero matrix and the
    adapter contributes nothing. The assertion is exact equality: a tolerance
    here would let a genuine perturbation — a mis-scaled ``alpha``, a
    non-zero ``lora_b``, a base weight accidentally copied instead of moved —
    hide under it.
    """

    def test_apply_lora_output_is_bit_identical(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0))
        assert torch.equal(base, predict(model, inputs))

    def test_dora_output_is_bit_identical(self) -> None:
        # DoRA rescales by m/‖W + ΔW‖ with m initialised to the base row norms,
        # so the rescale is exactly 1 and the identity survives the extra factor.
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0, use_dora=True))
        assert torch.equal(base, predict(model, inputs))

    def test_lora_b_is_zero_and_lora_a_is_not(self) -> None:
        # Both factors zero would be an identity too, but a permanently dead
        # one: the gradient of each factor is proportional to the other.
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj",)))
        for _, module in lora_modules(model):
            assert bool((module.lora_b == 0).all())
            assert float(module.lora_a.detach().abs().sum()) > 0.0

    def test_dora_magnitude_starts_at_the_base_row_norms(self) -> None:
        model = build_model()
        expected = model.blocks[0].attention.q_proj.weight.detach().pow(2).sum(dim=1)
        apply_lora(model, LoRAConfig(rank=4, use_dora=True, target_modules=("q_proj",)))
        adapted = model.get_submodule("blocks.0.attention.q_proj")
        assert adapted.lora_magnitude is not None
        torch.testing.assert_close(adapted.lora_magnitude, expected.sqrt())

    def test_a_perturbed_adapter_does_move_the_output(self) -> None:
        # The control for every identity assertion above: if this passed for a
        # broken adapter too, the identity tests would prove nothing.
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0))
        randomise_adapters(model)
        assert not torch.allclose(base, predict(model, inputs))

    def test_scaling_is_alpha_over_rank(self) -> None:
        # Raising rank at fixed alpha must leave the effective update magnitude
        # — and therefore the usable learning rate — unchanged.
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, alpha=32.0, target_modules=("q_proj",)))
        for _, module in lora_modules(model):
            assert module.scaling == pytest.approx(4.0)


class TestLoRAMerge:
    """Merging must reproduce the adapted model, and unmerging must undo it.

    Merging is what makes LoRA free at inference: the deployed model is an
    ordinary linear stack. It is only free if it is also *correct*, and the merge
    changes the order of operations — the unmerged path evaluates
    ``x·Wᵀ + ((x·Aᵀ)·Bᵀ)·s`` as two thin matmuls plus a residual, the merged path
    evaluates ``x·(W + s·B·A)ᵀ`` as one dense matmul. In real arithmetic these
    agree exactly; in fp32 they differ by reassociation rounding, which is why
    the bound below is ~1e-7 rather than zero.
    """

    def test_merge_reproduces_the_adapted_output(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0))
        randomise_adapters(model)
        adapted = predict(model, inputs)
        merge_lora(model)
        merged = predict(model, inputs)
        # fp32 reassociation only: the residual and the dense form sum the same
        # terms in a different order. Anything larger is a real merge error.
        torch.testing.assert_close(merged, adapted, rtol=0.0, atol=1e-6)

    def test_unmerge_restores_the_adapted_output(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0))
        randomise_adapters(model)
        adapted = predict(model, inputs)
        merge_lora(model)
        unmerge_lora(model)
        torch.testing.assert_close(predict(model, inputs), adapted, rtol=0.0, atol=1e-6)

    def test_dora_merge_is_exact(self) -> None:
        # DoRA rescales the whole weight, base included, so it cannot be written
        # as a residual: both the merged and unmerged forwards go through the
        # same dense F.linear, and the results agree bit for bit.
        model = build_model()
        inputs = make_inputs(model)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0, use_dora=True))
        randomise_adapters(model, std=0.02)
        adapted = predict(model, inputs)
        merge_lora(model)
        assert torch.equal(predict(model, inputs), adapted)

    def test_dora_unmerge_round_trips_within_rounding(self) -> None:
        """DoRA merge/unmerge must be an exact inverse.

        It was not, and the failure was silent and worth remembering. A DoRA
        merge writes ``m*(W+dW)/||W+dW||``, whose row norms are exactly ``m``,
        so recomputing ``||weight||/m`` from the merged weight gives exactly 1
        and the rescale step becomes a no-op — unmerge returned a weight a few
        percent off with no error raised. ``||W+dW||`` cannot be reconstructed
        from the merged weight, the magnitude and the delta, so merge now caches
        it. This test pins that.
        """
        model = build_model()
        inputs = make_inputs(model)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0, use_dora=True))
        randomise_adapters(model, std=0.02)
        adapted = predict(model, inputs)
        merge_lora(model)
        unmerge_lora(model)
        # The docstring on unmerge claims this is exact in real arithmetic and
        # accurate to rounding in fp32. It is neither: the observed error is
        # ~4% of the weight scale, not ~1e-7.
        torch.testing.assert_close(predict(model, inputs), adapted, rtol=0.0, atol=1e-5)

    def test_dora_merge_caches_the_row_norm_it_needs_to_unmerge(self) -> None:
        """The merged row norms equal the magnitude, so they must be cached.

        The minimal statement of why, independent of any model. A DoRA merge
        writes ``m*(W+dW)/||W+dW||``, whose row norms are *exactly* ``m``. Any
        unmerge that recomputes the scale from the merged weight therefore gets
        ``||weight||/m == 1`` and silently does nothing, returning a weight a
        few percent off with no error. ``||W+dW||`` has to be kept from merge
        time; it cannot be reconstructed afterwards.
        """
        torch.manual_seed(0)
        base = nn.Linear(8, 4, bias=False)
        module = LoRALinear.from_linear(base, rank=2, alpha=2.0, use_dora=True)
        with torch.no_grad():
            module.lora_b.normal_(0.0, 0.1)
        original = module.weight.detach().clone()
        module.merge()
        # The information the naive unmerge would have relied on is gone.
        torch.testing.assert_close(
            module.weight.detach().norm(dim=1), module.lora_magnitude.detach()
        )
        module.unmerge()
        torch.testing.assert_close(
            module.weight.detach(), original, rtol=0.0, atol=1e-5
        )

    def test_dora_falls_back_to_plain_lora_on_a_zero_weight(self) -> None:
        """A zero-initialised row has no magnitude, and DoRA is undefined there.

        Not hypothetical: every DiT-zero output projection starts at exactly
        zero, including the text cross-attention ``out_proj`` that a style
        fine-tune most wants to move. With ``m = 0`` the DoRA rescale is zero,
        so the merged row is identically zero and the adapter cannot influence
        it at all — the layer silently refuses to train. Those rows fall back to
        plain LoRA, which is what DoRA degenerates to when there is no
        pretrained magnitude to preserve.
        """
        torch.manual_seed(0)
        base = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            base.weight.zero_()
        module = LoRALinear.from_linear(base, rank=2, alpha=2.0, use_dora=True)
        with torch.no_grad():
            module.lora_b.normal_(0.0, 0.1)
        assert float(module.lora_magnitude.detach().abs().max()) == 0.0

        delta = module.lora_delta()
        # Plain-LoRA behaviour: the adapted weight is exactly the delta.
        torch.testing.assert_close(
            module.adapted_weight().detach(), delta.detach(), rtol=0.0, atol=1e-6
        )
        assert float(module.adapted_weight().detach().abs().max()) > 0.0, (
            "a zero-magnitude row must still be adaptable"
        )
        module.merge()
        module.unmerge()
        torch.testing.assert_close(
            module.weight.detach(), torch.zeros_like(module.weight), atol=1e-6, rtol=0.0
        )

    def test_merge_with_strip_leaves_plain_linears(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0))
        randomise_adapters(model)
        adapted = predict(model, inputs)
        merge_lora(model, strip=True)
        assert not list(lora_modules(model))
        projection = model.get_submodule("blocks.0.attention.q_proj")
        assert type(projection) is nn.Linear
        torch.testing.assert_close(predict(model, inputs), adapted, rtol=0.0, atol=1e-6)

    def test_double_merge_and_stray_unmerge_raise(self) -> None:
        # Silently tolerating either would produce a model whose weights are the
        # adapter folded in twice, or in zero times, with no way to tell.
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        _, module = next(iter(lora_modules(model)))
        with pytest.raises(RuntimeError, match="not merged"):
            module.unmerge()
        module.merge()
        with pytest.raises(RuntimeError, match="already merged"):
            module.merge()


class TestStateDictPaths:
    """Injection must not rename a single base-model checkpoint key.

    This is why ``LoRALinear`` subclasses ``nn.Linear`` rather than wrapping one.
    A wrapper turns ``blocks.7.attention.q_proj.weight`` into
    ``blocks.7.attention.q_proj.base_layer.weight``, which means the pretrained
    checkpoint no longer loads, the tensor-parallel plan no longer addresses
    anything, and a distributed checkpoint needs a bespoke conversion pass in
    both directions. Everything downstream of the adapter depends on this test.
    """

    def test_a_pretrained_checkpoint_still_loads_after_injection(self) -> None:
        model = build_model()
        pretrained = {name: value.clone() for name, value in model.state_dict().items()}
        apply_lora(model, LoRAConfig(rank=8, alpha=16.0))
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        # Nothing in the checkpoint is unrecognised, and everything the model now
        # wants beyond it is an adapter tensor the checkpoint never had.
        assert unexpected == []
        assert missing
        assert all(
            key.rpartition(".")[2] in ("lora_a", "lora_b", "lora_magnitude")
            for key in missing
        )

    def test_lora_linear_is_an_nn_linear(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        _, module = next(iter(lora_modules(model)))
        assert isinstance(module, nn.Linear)

    def test_the_base_weight_object_is_moved_not_copied(self) -> None:
        # Copying would double peak memory at injection time on a model already
        # sized to fill the device, and would silently detach the adapted module
        # from any sharding the original parameter carried.
        model = build_model()
        original = model.blocks[0].attention.q_proj.weight
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        assert model.get_submodule("blocks.0.attention.q_proj").weight is original

    def test_targets_are_addressed_by_the_contract_names(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4))
        names = {name for name, _ in lora_modules(model)}
        assert "blocks.0.attention.q_proj" in names
        assert "blocks.0.feed_forward.down_proj" in names


class TestAdapterSerialisation:
    """An adapter file must contain the adapter and nothing else.

    The whole distribution story is "a few megabytes a user drops next to any
    copy of the base checkpoint". That only holds if the file carries no base
    weights — which would make it gigabytes and tie it to one base — and enough
    metadata to rebuild the injection.
    """

    def test_round_trip_restores_the_adapter_exactly(self, tmp_path: Any) -> None:
        model = build_model()
        inputs = make_inputs(model)
        config = LoRAConfig(rank=8, alpha=16.0, target_modules=("q_proj", "v_proj"))
        apply_lora(model, config)
        randomise_adapters(model)
        adapted = predict(model, inputs)
        saved = {
            name: value.clone()
            for name, value in model.state_dict().items()
            if name.rpartition(".")[2].startswith("lora_")
        }

        path = save_adapter(tmp_path / "adapter.safetensors", model, config=config)
        with torch.no_grad():
            for parameter in lora_parameters(model):
                parameter.zero_()
        metadata = load_adapter(path, model)

        restored = model.state_dict()
        assert all(torch.equal(saved[key], restored[key]) for key in saved)
        assert metadata["format"] == "avgen-lora-v1"
        assert metadata["rank"] == "8"
        assert torch.equal(predict(model, inputs), adapted)

    def test_the_file_holds_only_adapter_tensors(self, tmp_path: Any) -> None:
        from safetensors import safe_open

        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj",)))
        path = save_adapter(tmp_path / "adapter.safetensors", model)
        with safe_open(str(path), framework="pt") as handle:
            keys = list(handle.keys())  # safe_open is not a Mapping
        assert keys
        assert all(
            key.rpartition(".")[2] in ("lora_a", "lora_b", "lora_magnitude")
            for key in keys
        )
        # A file carrying base weights would be three orders of magnitude larger
        # and would stop being portable across copies of the base checkpoint.
        assert not any(key.endswith(".weight") or key.endswith(".bias") for key in keys)

    def test_adapter_state_dict_matches_the_module_walk(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4, use_dora=True, target_modules=("q_proj",)))
        state = adapter_state_dict(model)
        assert len(state) == 3 * len(list(lora_modules(model)))

    def test_saving_a_merged_model_raises(self) -> None:
        # The factors of a merged adapter no longer describe the difference from
        # the saved base, so the file would silently be a no-op adapter.
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        merge_lora(model)
        with pytest.raises(RuntimeError, match="merged"):
            adapter_state_dict(model)

    def test_loading_a_missing_file_raises(self, tmp_path: Any) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        with pytest.raises(FileNotFoundError):
            load_adapter(tmp_path / "absent.safetensors", model)

    def test_a_rank_mismatch_is_an_error_not_a_silent_truncation(
        self, tmp_path: Any
    ) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj",)))
        path = save_adapter(tmp_path / "adapter.safetensors", model)
        other = build_model()
        apply_lora(other, LoRAConfig(rank=4, target_modules=("q_proj",)))
        with pytest.raises(ValueError, match="ranks disagree"):
            load_adapter(path, other)

    def test_strict_loading_rejects_an_unknown_key(self, tmp_path: Any) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj", "v_proj")))
        path = save_adapter(tmp_path / "adapter.safetensors", model)
        narrower = build_model()
        apply_lora(narrower, LoRAConfig(rank=8, target_modules=("q_proj",)))
        with pytest.raises(KeyError, match="apply_lora"):
            load_adapter(path, narrower)
        # Non-strict is the escape hatch, and it must still load what it can.
        assert load_adapter(path, narrower, strict=False)["format"] == "avgen-lora-v1"


class TestLoRAConfig:
    """Configuration errors must surface at construction, not at step 3000.

    An empty ``target_modules`` match is the expensive one: it trains zero
    parameters, the loss curve looks plausible because the frozen model still
    has a loss, and a cluster-week is spent before anyone notices.
    """

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rank": 0}, "rank must be"),
            ({"alpha": 0.0}, "alpha must be"),
            ({"dropout": 1.0}, "dropout must be"),
            ({"init_lora_weights": "orthogonal"}, "init_lora_weights must be"),
            ({"rank_pattern": (("q_proj", 0),)}, "must be a positive integer"),
            ({"alpha_pattern": (("q_proj", -1.0),)}, "must be positive"),
        ],
    )
    def test_invalid_fields_raise(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            LoRAConfig(**kwargs)

    def test_a_bare_string_target_is_rejected(self) -> None:
        # Iterating "q_proj" would yield its characters and match almost nothing
        # — or, worse, almost everything.
        with pytest.raises(TypeError, match="sequence of strings"):
            LoRAConfig(target_modules="q_proj")  # type: ignore[arg-type]

    def test_per_module_overrides_take_the_first_match(self) -> None:
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            rank_pattern=(("q_proj", 4), ("*_proj", 8)),
            alpha_pattern=(("q_proj", 1.0),),
        )
        assert config.rank_for("blocks.0.attention.q_proj") == 4
        assert config.rank_for("blocks.0.feed_forward.up_proj") == 8
        assert config.rank_for("patch_embed") == 16
        assert config.alpha_for("blocks.0.attention.q_proj") == pytest.approx(1.0)

    def test_exclusion_vetoes_a_target(self) -> None:
        config = LoRAConfig(
            target_modules=("q_proj", "v_proj"), exclude_modules=("blocks.0.*",)
        )
        assert config.selects("blocks.1.attention.q_proj")
        assert not config.selects("blocks.0.attention.q_proj")

    def test_an_override_actually_changes_the_injected_rank(self) -> None:
        model = build_model()
        apply_lora(
            model,
            LoRAConfig(
                rank=8,
                target_modules=("q_proj",),
                rank_pattern=(("blocks.0.attention.q_proj", 2),),
            ),
        )
        assert model.get_submodule("blocks.0.attention.q_proj").rank == 2
        assert model.get_submodule("blocks.1.attention.q_proj").rank == 8

    def test_an_empty_match_raises_and_names_what_exists(self) -> None:
        model = build_model()
        with pytest.raises(RuntimeError, match=r"no nn\.Linear matched"):
            apply_lora(model, LoRAConfig(rank=4, target_modules=("qkv_proj",)))

    def test_metadata_is_json_safe_strings(self) -> None:
        metadata = LoRAConfig(rank=8, alpha=16.0).to_metadata()
        assert all(isinstance(value, str) for value in metadata.values())


class TestFreezing:
    """Parameter selection decides whether a fine-tune fits, so it must be exact.

    The number that matters is the trainable *element* count: Adam holds two fp32
    moments per trainable element, and that allocation — not the parameters
    themselves — is usually what decides whether a job runs on one GPU or eight.
    Both selectors refuse an empty match, because "froze the entire model" and
    "froze nothing" are always typos rather than requests.
    """

    def test_freeze_all_freezes_every_tensor(self) -> None:
        model = build_model()
        count = freeze_all(model)
        assert count == sum(1 for _ in model.parameters())
        assert not any(p.requires_grad for p in model.parameters())

    def test_freeze_except_keeps_exactly_the_named_parameters(self) -> None:
        model = build_model()
        expected = [
            name
            for name, _ in model.named_parameters()
            if name.startswith("blocks.1.") or name == "final_proj.weight"
        ]
        summary = freeze_except(model, ("blocks.1.*", "final_proj.weight"))
        assert summary.trainable_tensors == len(expected)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        assert trainable == set(expected)

    def test_freeze_matching_is_non_destructive(self) -> None:
        # The blacklist form must compose with an earlier selection: a LoRA
        # injection followed by freeze_matching must not re-enable the base the
        # adapter just froze.
        model = build_model()
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        mark_only_lora_trainable(model)
        before = {name for name, p in model.named_parameters() if p.requires_grad}
        freeze_matching(model, ("blocks.0.attention.q_proj.lora_a",))
        after = {name for name, p in model.named_parameters() if p.requires_grad}
        assert after == before - {"blocks.0.attention.q_proj.lora_a"}

    def test_mark_only_lora_trainable_counts_the_adapters(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj", "v_proj")))
        count = mark_only_lora_trainable(model)
        adapters = lora_parameters(model)
        assert count == len(adapters)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) == len(adapters)
        assert all(any(p is a for a in adapters) for p in trainable)

    def test_mark_only_lora_trainable_honours_extra_patterns(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj",)))
        count = mark_only_lora_trainable(model, extra_trainable=("final_proj.weight",))
        assert count == len(lora_parameters(model)) + 1
        assert model.final_proj.weight.requires_grad

    def test_dora_magnitude_counts_as_trainable(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, use_dora=True, target_modules=("q_proj",)))
        count = mark_only_lora_trainable(model)
        assert count == 3 * len(list(lora_modules(model)))

    def test_summary_reports_global_elements_and_adam_state(self) -> None:
        model = build_model()
        apply_lora(model, LoRAConfig(rank=8, target_modules=("q_proj",)))
        mark_only_lora_trainable(model)
        summary = trainable_summary(model)
        assert summary.total == sum(p.numel() for p in model.parameters())
        # Two fp32 moments per trainable element: the number that decides
        # whether the fine-tune fits.
        assert summary.optimizer_state_bytes == summary.trainable * 8
        assert 0.0 < summary.percentage < 100.0
        assert "tensors" in summary.describe()

    def test_an_empty_match_raises_in_both_directions(self) -> None:
        model = build_model()
        with pytest.raises(ValueError, match="matched no parameter"):
            freeze_except(model, ("no_such_parameter",))
        with pytest.raises(ValueError, match="matched no parameter"):
            freeze_matching(model, ("no_such_parameter",))

    def test_a_bare_string_pattern_is_rejected(self) -> None:
        model = build_model()
        with pytest.raises(TypeError, match="sequence of strings"):
            freeze_except(model, "blocks.1")


class TestControlAdapter:
    """A control tower must contribute exactly zero before it is trained.

    ControlNet's zero-initialised output projection is what makes the adapted
    model reproduce the base bit-for-bit at step 0. The gradient with respect to
    that projection is *not* zero — it is the incoming gradient times the side
    tower's activation — so the layer learns immediately. Zero is an initial
    condition, not a dead branch, and both halves are asserted here.
    """

    def test_identity_at_initialisation(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        adapter = ControlAdapter(
            model, ControlAdapterConfig(control_width=8, num_blocks=2)
        )
        control = torch.randn((2, inputs.video.length, 8))
        with adapter.attach(model, control):
            attached = predict(model, inputs)
        assert torch.equal(base, attached)

    def test_a_trained_projection_does_inject(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        adapter = ControlAdapter(
            model, ControlAdapterConfig(control_width=8, num_blocks=2)
        )
        control = torch.randn((2, inputs.video.length, 8))
        with adapter.attach(model, control):
            predict(model, inputs)  # builds the lazily-sized projections
        assert adapter.projections is not None
        with torch.no_grad():
            for projection in adapter.projections:
                nn.init.normal_(projection.weight, std=0.05)
        with adapter.attach(model, control):
            assert not torch.allclose(base, predict(model, inputs))

    def test_detaching_restores_the_base_model(self) -> None:
        # Hooks that outlive their scope would silently contaminate every later
        # forward, including an evaluation pass that is meant to measure the base.
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        adapter = ControlAdapter(
            model, ControlAdapterConfig(control_width=8, num_blocks=2)
        )
        control = torch.randn((2, inputs.video.length, 8))
        handles = adapter.attach(model, control)
        predict(model, inputs)
        handles.remove()
        assert torch.equal(base, predict(model, inputs))
        handles.remove()  # idempotent
        assert len(handles) == 0

    def test_the_tower_copies_the_base_blocks_and_is_trainable(self) -> None:
        model = build_model()
        freeze_all(model)
        adapter = ControlAdapter(
            model, ControlAdapterConfig(control_width=8, num_blocks=2)
        )
        assert len(adapter.blocks) == 2
        assert all(p.requires_grad for p in adapter.blocks.parameters())
        # A copy, not a reference: training the tower must not touch the base.
        assert (
            adapter.blocks[0].attention.q_proj.weight
            is not model.blocks[0].attention.q_proj.weight
        )

    def test_injection_stride_thins_the_injection_points(self) -> None:
        model = build_model()
        adapter = ControlAdapter(
            model,
            ControlAdapterConfig(control_width=8, num_blocks=2, injection_stride=2),
        )
        assert adapter.injection_indices == (0,)

    def test_misaligned_control_tokens_raise(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        adapter = ControlAdapter(
            model, ControlAdapterConfig(control_width=8, num_blocks=2)
        )
        control = torch.randn((2, inputs.video.length + 1, 8))
        with (
            pytest.raises(ValueError, match="must align"),
            adapter.attach(model, control),
        ):
            predict(model, inputs)

    def test_a_tower_deeper_than_the_model_raises(self) -> None:
        model = build_model()
        with pytest.raises(ValueError, match="exceeds the base model"):
            ControlAdapter(model, ControlAdapterConfig(control_width=8, num_blocks=99))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"control_width": 0}, "control_width must be"),
            ({"control_width": 8, "num_blocks": 0}, "num_blocks must be"),
            ({"control_width": 8, "injection_stride": 0}, "injection_stride must be"),
            ({"control_width": 8, "conditioning_scale": -1.0}, "non-negative"),
        ],
    )
    def test_invalid_config_fields_raise(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            ControlAdapterConfig(**kwargs)


class TestIPAdapter:
    """Decoupled image cross-attention must also start as an exact identity.

    The image branch borrows the *frozen* queries — it calls the base model's own
    ``q_proj`` on the same hidden state the block's cross-attention received — so
    the two branches attend from one query space and their outputs are
    commensurable enough to add. The output projection is zero, so the sum is the
    text branch alone until training moves it.
    """

    def test_identity_at_initialisation(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        adapter = IPAdapter(model, IPAdapterConfig(image_width=8))
        with adapter.attach(model, torch.randn((2, 8))):
            assert torch.equal(base, predict(model, inputs))

    def test_a_trained_projection_does_inject(self) -> None:
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        adapter = IPAdapter(model, IPAdapterConfig(image_width=8))
        with torch.no_grad():
            for projection in adapter.out_proj:
                nn.init.normal_(projection.weight, std=0.05)
        with adapter.attach(model, torch.randn((2, 8))):
            assert not torch.allclose(base, predict(model, inputs))

    def test_targets_every_cross_attention_block(self) -> None:
        model = build_model()
        adapter = IPAdapter(model, IPAdapterConfig(image_width=8))
        assert adapter.target_names == (
            "blocks.0.cross_attention",
            "blocks.1.cross_attention",
        )
        assert adapter.width == model.config.width

    def test_head_count_is_read_from_the_base_attention(self) -> None:
        # Duplicating the head count in the adapter config is a number that can
        # disagree with the model; inference removes that failure mode.
        model = build_model()
        adapter = IPAdapter(model, IPAdapterConfig(image_width=8))
        assert adapter.heads == model.config.num_heads

    def test_image_tokens_accept_pooled_and_sequence_embeddings(self) -> None:
        model = build_model()
        adapter = IPAdapter(model, IPAdapterConfig(image_width=8, num_tokens=4))
        pooled = adapter.image_tokens(torch.randn((2, 8)))
        sequence = adapter.image_tokens(torch.randn((2, 5, 8)))
        assert pooled.shape == (2, 4, model.config.width)
        assert sequence.shape == pooled.shape

    def test_target_blocks_narrows_the_injection(self) -> None:
        model = build_model()
        adapter = IPAdapter(
            model, IPAdapterConfig(image_width=8, target_blocks=("blocks.1.*",))
        )
        assert adapter.target_names == ("blocks.1.cross_attention",)

    def test_a_model_without_cross_attention_raises(self) -> None:
        from dataclasses import replace

        config = replace(preset("tiny"), cross_attention=False, text_refiner_depth=0)
        torch.manual_seed(0)
        model = VideoDiT(config)
        with pytest.raises(ValueError, match="no block exposes"):
            IPAdapter(model, IPAdapterConfig(image_width=8))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"image_width": 0}, "image_width must be"),
            ({"image_width": 8, "num_tokens": 0}, "num_tokens must be"),
            ({"image_width": 8, "num_heads": 0}, "num_heads must be"),
            ({"image_width": 8, "scale": -0.5}, "scale must be non-negative"),
        ],
    )
    def test_invalid_config_fields_raise(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            IPAdapterConfig(**kwargs)


class TestStages:
    """A recipe is data, so every shipped one must actually build and apply.

    The point of ``FinetuneStage`` is that a user adapts a recipe by replacing a
    field in a config file rather than by copying a training script. That only
    holds if the shipped recipes are valid values — a stage whose patterns match
    nothing raises at ``apply_stage``, which is a broken recipe shipped as if it
    worked.
    """

    def test_every_registered_stage_applies_to_the_real_model(self) -> None:
        for name in list_stages():
            stage = get_stage(name)
            model = build_model()
            apply_stage(model, stage)
            summary = trainable_summary(model)
            assert summary.trainable_tensors > 0, f"{name} trains nothing"

    def test_standard_stages_is_a_copy_of_the_registry(self) -> None:
        stages = standard_stages()
        assert set(stages) == set(list_stages())
        stages.clear()
        # Mutating the returned mapping must not damage the shipped recipes.
        assert set(list_stages()) == set(standard_stages())

    def test_list_stages_is_sorted(self) -> None:
        # Sorted rather than insertion-ordered so the listing does not depend on
        # which module imported first.
        assert list(list_stages()) == sorted(list_stages())

    def test_the_lora_stage_injects_adapters_and_freezes_the_base(self) -> None:
        model = build_model()
        stage = get_stage("lora")
        assert stage.uses_adapter
        apply_stage(model, stage)
        adapters = lora_parameters(model)
        assert adapters
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) == len(adapters)

    def test_the_control_stage_freezes_the_base_tower(self) -> None:
        model = build_model()
        apply_stage(model, get_stage("control"))
        assert not any(p.requires_grad for p in model.blocks.parameters())

    def test_an_unknown_stage_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown stage"):
            get_stage("no_such_stage")

    def test_registering_a_duplicate_name_raises(self) -> None:
        stage = FinetuneStage(name="lora", description="a clashing recipe")
        with pytest.raises(KeyError, match="already registered"):
            register_stage(stage)

    def test_a_registered_stage_round_trips(self) -> None:
        stage = FinetuneStage(name="_test_stage", description="scratch recipe")
        try:
            register_stage(stage)
            assert get_stage("_test_stage") is stage
            assert "_test_stage" in list_stages()
        finally:
            from avgen.finetune import stages as stages_module

            stages_module._STAGES.pop("_test_stage", None)

    def test_with_overrides_leaves_the_shipped_recipe_untouched(self) -> None:
        stage = get_stage("full")
        modified = stage.with_overrides(learning_rate=5e-6)
        assert modified.learning_rate == pytest.approx(5e-6)
        assert get_stage("full").learning_rate == stage.learning_rate

    def test_to_dict_is_json_safe_and_normalises_the_mixture(self) -> None:
        import json

        stage = get_stage("duration_extension")
        payload = stage.to_dict()
        assert json.loads(json.dumps(payload))["name"] == "duration_extension"
        assert sum(payload["conditioning_mix"].values()) == pytest.approx(1.0)

    def test_stage_names_for_finds_recipes_covering_a_mode(self) -> None:
        assert "duration_extension" in stage_names_for(["CONTINUATION"])
        assert "full" not in stage_names_for(["CONTINUATION"])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"name": ""}, "name must be non-empty"),
            ({"learning_rate": 0.0}, "learning_rate must be"),
            ({"weight_decay": -1.0}, "weight_decay must be"),
            ({"warmup_steps": -1}, "warmup_steps must be"),
            ({"max_grad_norm": 0.0}, "max_grad_norm must be"),
            ({"ema_decay": 1.0}, "ema_decay must be"),
            ({"conditioning_mix": (("JOINT", 0.0),)}, "positive total weight"),
            ({"conditioning_mix": (("JOINT", -1.0),)}, "positive total weight"),
        ],
    )
    def test_invalid_stage_fields_raise(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        fields: dict[str, Any] = {"name": "x", "description": "y", **kwargs}
        with pytest.raises(ValueError, match=message):
            FinetuneStage(**fields)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"method": "magic"}, "method must be"),
            ({"student_steps": 0}, "student_steps must be"),
            ({"student_steps": 60}, "exceeds"),
        ],
    )
    def test_invalid_distillation_fields_raise(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            DistillationConfig(**kwargs)


class TestTimestepShift:
    """The resolution shift is one number that must be applied in two places.

    Training with a shifted noise distribution and sampling without it — or the
    reverse — is the classic cause of a high-resolution fine-tune that looks
    worse than the model it started from. These tests pin the map itself: it
    fixes both endpoints, is monotone, and moves mass toward high noise.
    """

    def test_shift_interpolates_between_the_anchors(self) -> None:
        assert resolution_shift(4096) == pytest.approx(1.0)
        assert resolution_shift(65536) == pytest.approx(3.0)
        midpoint = resolution_shift((4096 + 65536) // 2)
        assert 1.0 < midpoint < 3.0

    def test_shift_clamps_outside_the_anchors(self) -> None:
        # Extrapolating linearly past the calibrated points would produce a shift
        # nobody measured; clamping is the honest behaviour.
        assert resolution_shift(1) == pytest.approx(1.0)
        assert resolution_shift(10**9) == pytest.approx(3.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sequence_length": 0}, "sequence_length must be"),
            ({"sequence_length": 100, "max_length": 10}, "anchors must satisfy"),
            ({"sequence_length": 100, "base_shift": 0.5}, "at least 1.0"),
        ],
    )
    def test_invalid_anchors_raise(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            resolution_shift(**kwargs)

    def test_shift_timesteps_fixes_the_endpoints_and_raises_the_middle(self) -> None:
        levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        shifted = shift_timesteps(levels, 3.0)
        assert float(shifted[0]) == 0.0
        assert float(shifted[-1]) == pytest.approx(1.0)
        assert bool((shifted[1:-1] > levels[1:-1]).all())
        # Monotone, which is what makes it safe on a sampler step schedule too.
        assert bool((shifted[1:] > shifted[:-1]).all())

    def test_a_unit_shift_is_the_identity(self) -> None:
        levels = torch.rand(16)
        torch.testing.assert_close(shift_timesteps(levels, 1.0), levels)

    def test_a_non_positive_shift_raises(self) -> None:
        with pytest.raises(ValueError, match="shift must be positive"):
            shift_timesteps(torch.rand(4), 0.0)

    def test_timestep_plan_prefers_an_explicit_shift(self) -> None:
        plan = TimestepPlan(sampler="uniform", shift=2.0)
        assert plan.shift_for(65536) == pytest.approx(2.0)
        derived = TimestepPlan(sampler="uniform")
        assert derived.shift_for(65536) == pytest.approx(3.0)
        assert TimestepPlan(options=(("mean", 0.0),)).as_kwargs() == {"mean": 0.0}


class TestAdapterComposition:
    """LoRA and the hook-based adapters must be usable together.

    A depth-control tower and a style adapter are two files over one shared base;
    if attaching one disturbed the other the whole additive story collapses.
    """

    def test_lora_and_a_control_tower_compose_without_disturbing_the_base(
        self,
    ) -> None:
        model = build_model()
        inputs = make_inputs(model)
        base = predict(model, inputs)
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj", "v_proj")))
        adapter = ControlAdapter(
            model, ControlAdapterConfig(control_width=8, num_blocks=2)
        )
        control = torch.randn((2, inputs.video.length, 8))
        with adapter.attach(model, control):
            assert torch.equal(base, predict(model, inputs))

    def test_a_deep_copy_of_an_adapted_model_stays_adapted(self) -> None:
        # merge_lora is destructive, so the documented pattern is to merge a
        # copy. That only works if a copy survives the round trip.
        model = build_model()
        inputs = make_inputs(model)
        apply_lora(model, LoRAConfig(rank=4, target_modules=("q_proj",)))
        randomise_adapters(model)
        adapted = predict(model, inputs)
        clone = copy.deepcopy(model)
        merge_lora(clone)
        torch.testing.assert_close(predict(clone, inputs), adapted, rtol=0.0, atol=1e-6)
        assert torch.equal(predict(model, inputs), adapted)


def test_grid_patchifier_geometry_matches_the_model() -> None:
    """The adapters are tested against the patchifier the model was built for."""
    model = build_model()
    assert isinstance(model.patchifier, GridPatchifier)
    assert model.patchifier.patch_height == model.config.patch_height
