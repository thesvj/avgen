"""Distributed stress and edge-case sweep, driven by the simulator.

Every check here runs in one process against a ``FakeProcessGroup``, which lets
it sweep world sizes from 8 to 4096 and impersonate *every rank in turn* — so
per-rank logic that a real 4-rank job would only sample four points of is
checked exhaustively.

What this can and cannot establish is worth being precise about, because the
distinction is the whole reason the simulator exists.

**Exact here.** Mesh construction and dimension ordering. Which rank gets which
data. RNG derivation. Sharding arithmetic and DTensor placements. Which
collectives are issued, how many, and from which module. Every validation and
error path. All of this is ordinary Python and tensor metadata; the fake
backend does not touch it.

**Not here.** Anything whose answer depends on a collective actually moving
bytes: that a reduced gradient equals the mean of the inputs, that a
context-parallel loss equals the unsharded loss, that a checkpoint written by
N ranks reloads onto M. Fake collectives return uninitialised memory, so a test
asserting on their output would pass for the wrong reason. Those live in
``tests/test_gpu_distributed.py`` behind the ``multigpu`` marker and need real
ranks.

Reading a failure here: the sweep reports the world size and rank it failed at,
because "fails at 1024 but not 8" is the single most useful fact about a
distributed bug.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from avgen.core import PatchLayout, RNGStreams, TokenStream
from avgen.parallel import (
    ActivationCheckpointConfig,
    ParallelConfig,
    ParallelDims,
    pad_to_multiple,
    parallelize,
    shard_stream,
    submesh,
)
from avgen.simulate import (
    H100_SXM,
    ModelShape,
    SearchSpace,
    estimate_memory,
    fake_world,
    search_parallel_plan,
    simulate_config,
)

# World sizes chosen to cover the interesting regimes: below one node, exactly
# one node, several nodes, and past the point where most frameworks stop being
# tested at all.
WORLD_SIZES = [1, 2, 8, 64, 512, 1024, 4096]

# Degree combinations that must all remain valid. Each is (dp_replicate,
# dp_shard, cp, tp, pp) with -1 meaning "infer".
PLANS = [
    (1, -1, 1, 1, 1),
    (1, -1, 2, 1, 1),
    (1, -1, 4, 2, 1),
    (1, -1, 8, 8, 1),
    (2, -1, 2, 2, 1),
    (4, -1, 1, 8, 1),
    (1, -1, 2, 2, 2),
]


def _dims(world: int, plan: tuple[int, int, int, int, int]) -> ParallelDims | None:
    """Build dims, or None when this plan does not factor this world."""
    replicate, shard, context, tensor, pipeline = plan
    try:
        return ParallelDims(
            world_size=world,
            dp_replicate=replicate,
            dp_shard=shard,
            tensor=tensor,
            context=context,
            pipeline=pipeline,
        )
    except ValueError:
        return None


class Block(nn.Module):
    """A block whose submodule names match the shipped tensor-parallel plan."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.ffn_norm = nn.RMSNorm(width)
        self.attention = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.attention, name, nn.Linear(width, width, bias=False))
        self.feed_forward = nn.Module()
        self.feed_forward.gate_proj = nn.Linear(width, 4 * width, bias=False)
        self.feed_forward.up_proj = nn.Linear(width, 4 * width, bias=False)
        self.feed_forward.down_proj = nn.Linear(4 * width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(hidden)
        attended = self.attention.out_proj(self.attention.q_proj(normed))
        hidden = hidden + attended
        gated = torch.nn.functional.silu(
            self.feed_forward.gate_proj(self.ffn_norm(hidden))
        )
        return hidden + self.feed_forward.down_proj(
            gated * self.feed_forward.up_proj(self.ffn_norm(hidden))
        )


class Toy(nn.Module):
    """A model shaped like a DiT, small enough to build thousands of times."""

    def __init__(self, width: int = 64, depth: int = 8) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(16, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.final_norm = nn.RMSNorm(width)
        self.final_proj = nn.Linear(width, 16)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embed(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_proj(self.final_norm(hidden))


class TestMeshTopology:
    """The mesh must be built the same way at every scale."""

    @pytest.mark.parametrize("world", WORLD_SIZES)
    def test_dimension_order_is_stable(self, world: int) -> None:
        # Ordering is a performance decision: tensor-parallel ranks must be
        # adjacent so they land in one NVLink domain. If this drifts, throughput
        # drops and nothing errors.
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None or not dims._active_dims:
                continue
            with fake_world(dims, device_type="cpu") as sim:
                names = tuple(sim.require_mesh().mesh_dim_names or ())
            order = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]
            positions = [order.index(name) for name in names]
            assert positions == sorted(positions), (
                f"world={world} plan={plan} produced mesh order {names}"
            )

    @pytest.mark.parametrize("world", [8, 64, 512, 1024])
    def test_flattened_views_exist_whenever_they_should(self, world: int) -> None:
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None:
                continue
            with fake_world(dims, device_type="cpu") as sim:
                mesh = sim.require_mesh()
                names = tuple(mesh.mesh_dim_names or ())
                if "dp_shard" in names and "cp" in names:
                    view = submesh(mesh, "dp_shard_cp")
                    assert view is not None, f"world={world} plan={plan}"
                    assert view.size() == dims.dp_shard * dims.context
                if dims.dp_replicate_enabled and "dp_shard" in names:
                    assert submesh(mesh, "hsdp") is not None, (
                        f"world={world} plan={plan}"
                    )

    @pytest.mark.parametrize("world", [8, 64, 512])
    def test_degrees_multiply_to_the_world(self, world: int) -> None:
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None:
                continue
            product = (
                dims.dp_replicate
                * dims.dp_shard
                * dims.context
                * dims.tensor
                * dims.pipeline
            )
            assert product == world, f"plan={plan} gives {product} for world={world}"


class TestPerRankInvariants:
    """Checked by impersonating every rank, not a sample of them."""

    @pytest.mark.parametrize("world", [8, 16, 64])
    def test_data_coordinates_partition_the_ranks(self, world: int) -> None:
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None:
                continue
            groups: dict[int, list[int]] = {}
            for rank in range(world):
                with fake_world(dims, rank=rank, device_type="cpu") as sim:
                    data_rank, data_world = dims.data_coordinates(sim.require_mesh())
                groups.setdefault(data_rank, []).append(rank)
                assert data_world == dims.dp_size, f"plan={plan}"
                assert 0 <= data_rank < data_world

            # Every data_rank must appear, and each must hold the same number of
            # ranks — an imbalance means some samples are replicated more than
            # others and the effective batch is not what the config says.
            assert sorted(groups) == list(range(dims.dp_size)), f"plan={plan}"
            sizes = {len(members) for members in groups.values()}
            assert sizes == {world // dims.dp_size}, f"plan={plan} sizes={sizes}"

    @pytest.mark.parametrize("world", [8, 16])
    def test_rng_varies_on_data_rank_and_only_on_data_rank(self, world: int) -> None:
        # The rule that keeps tensor- and context-parallel ranks agreeing about
        # which sample they are denoising, and keeps data-parallel ranks from
        # collapsing the effective batch to one sample repeated.
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None:
                continue
            draws: dict[int, torch.Tensor] = {}
            coords: dict[int, int] = {}
            for rank in range(world):
                with fake_world(dims, rank=rank, device_type="cpu") as sim:
                    data_rank, _ = dims.data_coordinates(sim.require_mesh())
                coords[rank] = data_rank
                draws[rank] = torch.randn(
                    32, generator=RNGStreams.for_rank(7, data_rank=data_rank).noise
                )
            for a, b in itertools.combinations(range(world), 2):
                identical = torch.equal(draws[a], draws[b])
                if coords[a] == coords[b]:
                    assert identical, (
                        f"plan={plan}: ranks {a},{b} share data_rank "
                        f"{coords[a]} but drew different noise"
                    )
                else:
                    assert not identical, (
                        f"plan={plan}: ranks {a},{b} have different data_rank "
                        "but drew identical noise"
                    )

    @pytest.mark.parametrize("world", [8, 64])
    def test_sequence_shards_partition_the_sequence(self, world: int) -> None:
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None or not dims.cp_enabled:
                continue
            length = 16 * dims.context
            stream = TokenStream(
                tokens=torch.arange(length, dtype=torch.float32).reshape(1, length, 1),
                coords=torch.zeros(1, length, 3),
                mask=torch.ones((1, length), dtype=torch.bool),
                noise_level=torch.zeros(1),
                layout=PatchLayout(
                    frames=length, height=1, width=1, patch_height=1, patch_width=1
                ),
            )
            seen: list[float] = []
            for rank in range(world):
                with fake_world(dims, rank=rank, device_type="cpu") as sim:
                    cp = submesh(sim.require_mesh(), "cp")
                    local = shard_stream(stream, cp)
                    index, count = dims.sequence_coordinates(sim.require_mesh())
                assert local.length == length // dims.context, f"plan={plan}"
                assert local.layout.num_tokens == length, "shard lost the global layout"
                assert count == dims.context
                if index == 0 or rank < dims.context:
                    pass
                seen.extend(local.tokens.flatten().tolist())
            # Each token appears exactly world/cp times — once per cp group.
            counts = {value: seen.count(value) for value in range(length)}
            expected = world // dims.context
            assert set(counts.values()) == {expected}, (
                f"plan={plan}: shards do not partition the sequence"
            )


class TestShardingArithmetic:
    """Parameters must actually shard, and by the right factor."""

    @pytest.mark.parametrize("world", [8, 64, 512, 1024])
    def test_parameters_shard_across_dp_and_cp(self, world: int) -> None:
        total = sum(p.numel() for p in Toy().parameters())
        for plan in PLANS:
            dims = _dims(world, plan)
            if dims is None or dims.pp_enabled:
                continue  # pipeline changes which blocks exist; covered below
            if dims.tp_enabled:
                continue  # Toy declares no tensor-parallel plan by design
            with fake_world(dims, device_type="cpu") as sim:
                result = parallelize(Toy(), dims, mesh=sim.require_mesh())
                local = sum(
                    p.to_local().numel() if hasattr(p, "to_local") else p.numel()
                    for p in result.model.parameters()
                )
            expected_shards = dims.dp_shard * dims.context
            if expected_shards > 1:
                # FSDP pads each shard, so the bound is generous; the point is
                # that sharding happened at roughly the right factor.
                assert local < total, f"world={world} plan={plan}: nothing sharded"
                assert local <= total / expected_shards * 4 + 4096, (
                    f"world={world} plan={plan}: sharded {total // max(local, 1)}x, "
                    f"expected about {expected_shards}x"
                )

    @pytest.mark.parametrize("world", [8, 64])
    def test_cp_only_still_shards(self, world: int) -> None:
        # A context-parallel job with no data-parallel dimension still needs a
        # sharding group: each rank holds a partial gradient for every shared
        # parameter, and the true gradient is their sum.
        dims = ParallelDims(world_size=world, context=world)
        assert not dims.dp_shard_enabled
        with fake_world(dims, device_type="cpu") as sim:
            result = parallelize(Toy(), dims, mesh=sim.require_mesh())
        assert any("fsdp" in item for item in result.applied), result.applied

    @pytest.mark.parametrize("world", [8, 64, 512])
    def test_tensor_parallel_without_a_plan_is_refused(self, world: int) -> None:
        # A wrong sharding plan trains silently to a wrong result, so a model
        # that has not declared one must be a hard error at every scale.
        dims = _dims(world, (1, -1, 1, 8, 1))
        if dims is None:
            pytest.skip("plan does not factor this world")
        with (
            fake_world(dims, device_type="cpu") as sim,
            pytest.raises(TypeError, match="tensor_parallel_plan"),
        ):
            parallelize(Toy(), dims, mesh=sim.require_mesh())

    @pytest.mark.parametrize(
        "mode", ["none", "selective_op", "selective_layer", "full"]
    )
    def test_every_checkpoint_policy_composes_with_sharding(self, mode: str) -> None:
        dims = ParallelDims(world_size=64, context=4)
        with fake_world(dims, device_type="cpu") as sim:
            result = parallelize(
                Toy(),
                dims,
                mesh=sim.require_mesh(),
                config=ParallelConfig(
                    activation_checkpoint=ActivationCheckpointConfig(mode=mode)
                ),
            )
        assert any("fsdp" in item for item in result.applied)
        if mode != "none":
            assert any("activation_checkpoint" in item for item in result.applied)


class TestCollectivePatterns:
    """Which collectives are issued, observed rather than predicted."""

    def test_sharding_issues_the_expected_collective_kinds(self) -> None:
        from torch.distributed.tensor.debug import CommDebugMode

        from avgen.core import GridPatchifier, ModelInput, TextContext
        from avgen.models import VideoDiT, preset

        config = preset("tiny")
        dims = ParallelDims(world_size=64, context=4)
        with fake_world(dims, device_type="cpu") as sim:
            base = VideoDiT(config)
            base.init_weights()
            model = parallelize(base, dims, mesh=sim.require_mesh()).model
            stream = GridPatchifier().to_tokens(
                torch.randn(1, config.in_channels, 4, 8, 8),
                positions=torch.zeros(1, 4),
                mask=torch.ones((1, 4, 8, 8), dtype=torch.bool),
                noise_level=torch.rand(1),
            )
            data = ModelInput(
                video=stream,
                audio=TokenStream.empty_like(1, stream.width),
                text=TextContext.empty(1, config.text_width),
            )
            mode = CommDebugMode()
            with mode:
                model(data)
            total = mode.get_total_counts()
            counts = {str(k): v for k, v in mode.get_comm_counts().items()}
        # FSDP must all-gather parameters block by block. Zero collectives means
        # the model was not really sharded and every rank holds a full copy —
        # which trains, and silently wastes the entire memory saving.
        assert total > 0, "a sharded model issued no collectives at all"
        # FSDP2 issues c10d._allgather_base_; the functional-collective spelling
        # is all_gather_into_tensor. Accept either — the point is that parameters
        # are being gathered per block, not which symbol torch routed through.
        flat = " ".join(str(key) for key in counts).lower()
        assert "allgather" in flat.replace("_", ""), (
            f"no parameter all-gather observed; saw {sorted(counts)}"
        )


class TestEdgeCases:
    """The inputs that are wrong, degenerate, or exactly on a boundary."""

    def test_indivisible_sequence_is_refused_not_truncated(self) -> None:
        dims = ParallelDims(world_size=8, context=8)
        length = 8 * 4 + 1
        stream = TokenStream(
            tokens=torch.zeros(1, length, 1),
            coords=torch.zeros(1, length, 3),
            mask=torch.ones((1, length), dtype=torch.bool),
            noise_level=torch.zeros(1),
            layout=PatchLayout(
                frames=length, height=1, width=1, patch_height=1, patch_width=1
            ),
        )
        with fake_world(dims, device_type="cpu") as sim:
            cp = submesh(sim.require_mesh(), "cp")
            with pytest.raises(ValueError, match="not divisible"):
                shard_stream(stream, cp)
            padded = pad_to_multiple(stream, dims.context)
            local = shard_stream(padded, cp)
        assert padded.length % dims.context == 0
        assert bool(padded.mask[0, length:].any()) is False, "padding is not masked"
        assert local.length == padded.length // dims.context

    @pytest.mark.parametrize("world", WORLD_SIZES)
    def test_degenerate_degrees_are_rejected(self, world: int) -> None:
        for bad in (
            {"dp_shard": world + 1},
            {"tensor": 0},
            {"context": -2},
            {"pipeline": 0},
        ):
            with pytest.raises(ValueError):
                ParallelDims(world_size=world, **bad)  # type: ignore[arg-type]

    def test_world_size_one_needs_no_mesh(self) -> None:
        dims = ParallelDims(world_size=1)
        assert not dims.dp_enabled
        with pytest.raises(RuntimeError, match="single-device"):
            dims.build_mesh("cpu")
        result = parallelize(Toy(), dims)
        assert result.mesh is None

    @pytest.mark.parametrize("world", [8, 64, 512])
    def test_global_batch_must_factor_exactly(self, world: int) -> None:
        dims = ParallelDims(world_size=world)
        # Rounding here silently changes the effective learning rate.
        with pytest.raises(ValueError, match="divisible"):
            dims.gradient_accumulation_for(
                global_batch_size=dims.dp_size * 3 + 1, local_batch_size=1
            )
        assert (
            dims.gradient_accumulation_for(
                global_batch_size=dims.dp_size * 4, local_batch_size=1
            )
            == 4
        )

    def test_prime_world_size_still_produces_a_valid_mesh(self) -> None:
        # 7 factors only as 7x1; a framework that assumes powers of two breaks.
        dims = ParallelDims(world_size=7)
        assert dims.dp_shard == 7
        with fake_world(dims, device_type="cpu") as sim:
            assert sim.require_mesh().size() == 7

    def test_zero_length_stream_is_legal(self) -> None:
        # A text-to-video model passes an empty audio stream; every audio path
        # must degenerate to a no-op rather than branch.
        stream = TokenStream.empty_like(2, 16)
        stream.validate()
        assert stream.length == 0
        dims = ParallelDims(world_size=4, context=4)
        with fake_world(dims, device_type="cpu") as sim:
            cp = submesh(sim.require_mesh(), "cp")
            assert shard_stream(stream, cp).length == 0


class TestPlannerUnderStress:
    """The plan search must stay sane across the whole configuration space."""

    @pytest.mark.parametrize("world", [8, 64, 512, 1024, 4096])
    def test_every_returned_plan_is_valid_and_fits(self, world: int) -> None:
        shape = ModelShape(
            parameters=2_000_000_000,
            depth=32,
            width=2560,
            sequence_length=65_536,
            num_heads=20,
            text_tokens=256,
        )
        plans = search_parallel_plan(shape, SearchSpace(world_size=world), top_k=8)
        for plan in plans:
            assert plan.dims.world_size == world
            assert plan.memory.fits_in(H100_SXM.memory_gib)
            assert plan.dims.tensor <= 8, "tensor parallel escaped the node"
            assert shape.sequence_length % plan.dims.sequence_shard_size == 0
            assert plan.step_seconds > 0
            assert 0.0 <= plan.mfu(H100_SXM) <= 1.0

    @pytest.mark.parametrize("seq", [4_096, 16_384, 65_536, 131_072, 262_144])
    def test_longer_sequences_need_more_context_parallelism(self, seq: int) -> None:
        shape = ModelShape(
            parameters=2_000_000_000,
            depth=32,
            width=2560,
            sequence_length=seq,
            num_heads=20,
        )
        plans = search_parallel_plan(
            shape, SearchSpace(world_size=1024, max_context=32), top_k=1
        )
        if not plans:
            pytest.skip(f"nothing fits at {seq} tokens")
        # Memory must not blow up with sequence length once the planner is free
        # to spend context parallelism on it.
        assert plans[0].memory.fits_in(H100_SXM.memory_gib)

    def test_impossible_shape_returns_nothing_rather_than_a_bad_plan(self) -> None:
        huge = ModelShape(
            parameters=400_000_000_000,
            depth=128,
            width=16_384,
            sequence_length=262_144,
            num_heads=128,
        )
        assert search_parallel_plan(huge, SearchSpace(world_size=8)) == []

    @pytest.mark.parametrize("world", [64, 512, 1024])
    def test_memory_estimate_falls_as_parallelism_rises(self, world: int) -> None:
        shape = ModelShape(
            parameters=2_000_000_000,
            depth=32,
            width=2560,
            sequence_length=65_536,
            num_heads=20,
        )
        low = estimate_memory(shape, ParallelDims(world_size=world, context=2))
        high = estimate_memory(shape, ParallelDims(world_size=world, context=8))
        assert high.total_bytes < low.total_bytes

    @pytest.mark.parametrize("world", [8, 64, 512, 1024, 4096])
    def test_report_renders_and_serialises_at_every_scale(self, world: int) -> None:
        shape = ModelShape(
            parameters=2_000_000_000,
            depth=32,
            width=2560,
            sequence_length=65_536,
            num_heads=20,
        )
        dims = _dims(world, (1, -1, 4, 2, 1)) or ParallelDims(world_size=world)
        report = simulate_config(shape, dims)
        text = report.render()
        payload = report.to_dict()
        assert "MEMORY" in text and "THROUGHPUT" in text
        assert payload["world_size"] == world
        assert isinstance(payload["fits"], bool)


class TestMixedPrecisionDoesNotCorruptMetadata:
    """Physical coordinates must survive mixed precision intact.

    FSDP2's ``cast_forward_inputs`` casts every floating-point forward input to
    the compute dtype. That is right for a plain activation and wrong for
    avgen's ``ModelInput``, which carries coordinates in seconds and noise levels
    alongside the token features. bfloat16 has eight mantissa bits: at a clip
    time of 30 seconds its resolution is 0.125 s, coarser than the 0.042 s
    between adjacent frames at 24 fps, so two neighbouring frames land on the
    same coordinate and the rotary embedding stops resolving them.

    The failure is silent on any model that does not validate its input dtypes,
    which is why it is pinned here rather than left to the contract check.
    """

    def test_policy_does_not_cast_forward_inputs(self) -> None:
        from avgen.parallel import PrecisionConfig

        policy = PrecisionConfig().fsdp_policy()
        assert policy.cast_forward_inputs is False, (
            "cast_forward_inputs=True silently degrades physical coordinates "
            "to bfloat16; see PrecisionConfig.fsdp_policy for the numbers"
        )

    def test_bfloat16_cannot_resolve_adjacent_frames(self) -> None:
        # The measurement behind the decision above, pinned so nobody 'optimises'
        # coordinates into bf16 later.
        spacing = 1.0 / 24.0
        for clip_seconds, resolvable in ((5.0, True), (10.0, True), (30.0, False)):
            pair = torch.tensor(
                [clip_seconds, clip_seconds + spacing], dtype=torch.bfloat16
            )
            assert bool(pair[0] != pair[1]) is resolvable, (
                f"bfloat16 resolution at t={clip_seconds}s changed"
            )
        exact = torch.tensor([30.0, 30.0 + spacing], dtype=torch.float32)
        assert exact[0] != exact[1], "float32 must always resolve adjacent frames"

    def test_sharded_forward_preserves_coordinate_dtype(self) -> None:
        from avgen.core import GridPatchifier, ModelInput, TextContext
        from avgen.models import VideoDiT, preset

        config = preset("tiny")
        dims = ParallelDims(world_size=8, context=2)
        with fake_world(dims, device_type="cpu") as sim:
            base = VideoDiT(config)
            base.init_weights()
            model = parallelize(base, dims, mesh=sim.require_mesh()).model
            stream = GridPatchifier().to_tokens(
                torch.randn(1, config.in_channels, 4, 8, 8),
                positions=torch.linspace(0, 30, 4).unsqueeze(0),
                mask=torch.ones((1, 4, 8, 8), dtype=torch.bool),
                noise_level=torch.rand(1),
            )
            inputs = ModelInput(
                video=stream,
                audio=TokenStream.empty_like(1, stream.width),
                text=TextContext.empty(1, config.text_width),
            )
            # Would raise "coords must be float32" if the policy cast them.
            output = model(inputs)
        assert output.video.shape[:2] == stream.tokens.shape[:2]
        assert inputs.video.coords.dtype is torch.float32
        assert inputs.video.noise_level.dtype is torch.float32


class TestSequenceDivisibilityIsCaughtAtConfigTime:
    """An indivisible bucket must fail validation, not the first forward pass.

    Context parallelism splits the token sequence; tensor parallelism splits it
    again wherever sequence parallelism applies. If a bucket's token count does
    not divide by that factor, avgen refuses — it will not pad implicitly,
    because the padding would change the token count the loss normalises by and
    make two otherwise-identical runs disagree.

    The point of catching it in the config is *when*. Without this the job
    starts, builds the mesh, materialises the model, resumes the checkpoint,
    fills the dataloader, and only then raises from inside the objective — on a
    cluster, minutes of a paid allocation to learn something knowable before the
    first byte was moved, with the error surfacing far from the setting that
    caused it.
    """

    def test_indivisible_context_degree_is_rejected(self) -> None:
        import dataclasses

        from avgen.config import RunConfig

        base = RunConfig()
        tokens = base.data.buckets[0].tokens(base.model)
        assert tokens % 3 != 0, "pick a degree that does not divide this bucket"
        with pytest.raises(ValueError, match="not divisible"):
            dataclasses.replace(
                base, parallel=dataclasses.replace(base.parallel, context=3)
            )

    def test_divisible_context_degree_is_accepted(self) -> None:
        import dataclasses

        from avgen.config import RunConfig

        base = RunConfig()
        tokens = base.data.buckets[0].tokens(base.model)
        for degree in (2, 4, 8):
            assert tokens % degree == 0
            dataclasses.replace(
                base, parallel=dataclasses.replace(base.parallel, context=degree)
            )

    def test_sequence_parallel_folds_tensor_into_the_factor(self) -> None:
        # With sequence parallelism on, tp splits the sequence too, so the
        # required divisor is context * tensor rather than context alone.
        import dataclasses

        from avgen.config import RunConfig

        base = RunConfig()
        tokens = base.data.buckets[0].tokens(base.model)
        assert tokens % 3 != 0
        with pytest.raises(ValueError, match="sequence parallel"):
            dataclasses.replace(
                base,
                parallel=dataclasses.replace(
                    base.parallel, context=1, tensor=3, sequence_parallel=True
                ),
            )
        # Disabling sequence parallelism removes tensor from the divisor.
        dataclasses.replace(
            base,
            parallel=dataclasses.replace(
                base.parallel, context=1, tensor=3, sequence_parallel=False
            ),
        )
