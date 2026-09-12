"""Tests for the distributed layer.

Most of these run without any process group at all, because the parts that are
easiest to get wrong — the factorisation arithmetic, which rank gets which data,
where a shard boundary falls — are pure functions. The ones that do need a world
use a fake process group, so a 1024-rank mesh is tested on one CPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from avgen.core import PatchLayout, TokenStream
from avgen.parallel import (
    ActivationCheckpointConfig,
    ParallelConfig,
    ParallelDims,
    PipelineConfig,
    PrecisionConfig,
    apply_activation_checkpointing,
    balanced_split_points,
    pad_to_multiple,
    parallelize,
    shard_stream,
    sharded_length,
    standard_block_plan,
    unwrap_model,
)
from avgen.simulate import fake_world


class TestParallelDims:
    def test_infers_dp_shard_from_the_world(self) -> None:
        dims = ParallelDims(world_size=1024, tensor=8, context=4)
        assert dims.dp_shard == 32
        assert dims.dp_size == 32

    def test_rejects_a_factorisation_that_does_not_multiply(self) -> None:
        with pytest.raises(ValueError, match="must multiply to world_size"):
            ParallelDims(world_size=100, dp_shard=8, tensor=8)

    def test_rejects_an_uninferable_dp_shard(self) -> None:
        with pytest.raises(ValueError, match="cannot infer dp_shard"):
            ParallelDims(world_size=100, tensor=8)

    def test_sequence_shard_size_folds_tp_and_cp(self) -> None:
        # Sequence parallelism means tensor parallelism also splits the
        # sequence, so both axes reduce per-rank tokens.
        dims = ParallelDims(world_size=64, tensor=4, context=4)
        assert dims.sequence_shard_size == 16

    def test_gradient_accumulation_must_divide_exactly(self) -> None:
        dims = ParallelDims(world_size=8)
        assert (
            dims.gradient_accumulation_for(global_batch_size=64, local_batch_size=2)
            == 4
        )
        # Rounding here would silently change the effective learning rate.
        with pytest.raises(ValueError, match="must be divisible"):
            dims.gradient_accumulation_for(global_batch_size=65, local_batch_size=2)

    def test_describe_is_readable(self) -> None:
        assert "cp=4" in ParallelDims(world_size=32, context=4).describe()
        assert "single-device" in ParallelDims(world_size=1).describe()


class TestMesh:
    def test_mesh_dimension_order_places_tp_innermost(self) -> None:
        # TP communicates twice per block on the critical path, so its ranks
        # must be adjacent and land inside one NVLink domain.
        dims = ParallelDims(world_size=64, dp_shard=4, context=2, tensor=8)
        with fake_world(dims, device_type="cpu") as world:
            names = tuple(world.require_mesh().mesh_dim_names or ())
        assert names[-1] == "tp"
        assert names.index("dp_shard") < names.index("cp") < names.index("tp")

    def test_degenerate_dimensions_are_omitted(self) -> None:
        dims = ParallelDims(world_size=8)
        with fake_world(dims, device_type="cpu") as world:
            assert tuple(world.require_mesh().mesh_dim_names or ()) == ("dp_shard",)

    def test_flattened_views_are_registered(self) -> None:
        from avgen.parallel import submesh

        dims = ParallelDims(world_size=64, dp_shard=8, context=4, tensor=2)
        with fake_world(dims, device_type="cpu") as world:
            mesh = world.require_mesh()
            assert submesh(mesh, "dp_shard_cp") is not None
            assert submesh(mesh, "dp_shard_cp").size() == 32
            assert submesh(mesh, "dp_cp") is not None
            assert submesh(mesh, "nonexistent") is None

    def test_data_coordinates_ignore_cp_and_tp(self) -> None:
        # The load-bearing rule: ranks holding shards of the same sample must
        # share a data_rank, or the effective batch silently shrinks.
        dims = ParallelDims(world_size=16, dp_shard=2, context=4, tensor=2)
        ranks = {}
        for rank in range(16):
            with fake_world(dims, rank=rank, device_type="cpu") as world:
                data_rank, data_world = dims.data_coordinates(world.require_mesh())
                ranks[rank] = data_rank
        assert data_world == 2
        assert set(ranks.values()) == {0, 1}
        # Exactly world_size / dp_size ranks share each data_rank.
        assert sum(1 for value in ranks.values() if value == 0) == 8


class TestContextParallel:
    def make_stream(self, length: int) -> TokenStream:
        return TokenStream(
            tokens=torch.arange(length, dtype=torch.float32).reshape(1, length, 1),
            coords=torch.zeros(1, length, 3),
            mask=torch.ones((1, length), dtype=torch.bool),
            noise_level=torch.zeros(1),
            layout=PatchLayout(
                frames=length, height=1, width=1, patch_height=1, patch_width=1
            ),
        )

    def test_sharded_length_refuses_a_ragged_split(self) -> None:
        assert sharded_length(64, 8) == 8
        with pytest.raises(ValueError, match="not divisible"):
            sharded_length(65, 8)

    def test_pad_to_multiple_masks_the_padding(self) -> None:
        padded = pad_to_multiple(self.make_stream(10), 4)
        assert padded.length == 12
        assert padded.mask[0, 10:].tolist() == [False, False]
        assert padded.mask[0, :10].all()

    def test_pad_is_a_no_op_when_already_aligned(self) -> None:
        stream = self.make_stream(8)
        assert pad_to_multiple(stream, 4) is stream

    def test_shard_takes_a_contiguous_slice(self) -> None:
        # Bidirectional attention means contiguous shards are load-balanced, so
        # there is no need for the zigzag ordering causal models require.
        dims = ParallelDims(world_size=4, context=4)
        stream = self.make_stream(16)
        seen = []
        for rank in range(4):
            with fake_world(dims, rank=rank, device_type="cpu") as world:
                local = shard_stream(stream, world.require_mesh()["cp"])
                seen.append(local.tokens.flatten().tolist())
        assert seen[0] == [0.0, 1.0, 2.0, 3.0]
        assert seen[3] == [12.0, 13.0, 14.0, 15.0]

    def test_shard_preserves_the_global_layout(self) -> None:
        dims = ParallelDims(world_size=4, context=4)
        stream = self.make_stream(16)
        with fake_world(dims, device_type="cpu") as world:
            local = shard_stream(stream, world.require_mesh()["cp"])
        assert local.length == 4
        assert local.layout.num_tokens == 16  # global geometry survives


class Block(nn.Module):
    """A block whose submodule names match ``standard_block_plan``."""

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
        return hidden + self.feed_forward.down_proj(
            torch.nn.functional.silu(self.feed_forward.gate_proj(hidden))
            * self.feed_forward.up_proj(hidden)
        )


class Toy(nn.Module):
    def __init__(self, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(8, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.final_norm = nn.RMSNorm(width)
        self.final_proj = nn.Linear(width, 8)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embed(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_proj(self.final_norm(hidden))


class TestActivationCheckpointing:
    def test_wrapping_preserves_numerics(self) -> None:
        torch.manual_seed(0)
        model = Toy()
        data = torch.randn(2, 6, 8)
        expected = model(data)
        apply_activation_checkpointing(
            model, ActivationCheckpointConfig(mode="selective_op")
        )
        assert torch.allclose(model(data), expected, atol=1e-6)

    def test_selective_layer_wraps_only_the_interval(self) -> None:
        model = Toy(depth=4)
        apply_activation_checkpointing(
            model, ActivationCheckpointConfig(mode="selective_layer", layer_interval=2)
        )
        wrapped = [type(b).__name__ for b in model.blocks]
        assert wrapped.count("CheckpointWrapper") == 2

    def test_none_is_a_no_op(self) -> None:
        model = Toy()
        apply_activation_checkpointing(model, ActivationCheckpointConfig(mode="none"))
        assert all(isinstance(b, Block) for b in model.blocks)

    def test_missing_block_list_is_a_clear_error(self) -> None:
        with pytest.raises(AttributeError, match="block_attribute"):
            apply_activation_checkpointing(
                nn.Linear(2, 2), ActivationCheckpointConfig(mode="full")
            )


class TestParallelize:
    def test_single_device_builds_no_mesh(self) -> None:
        # Activation checkpointing still applies — it is a memory/compute trade
        # that is useful on one GPU too — but no process group is created.
        result = parallelize(Toy(), ParallelDims(world_size=1))
        assert result.mesh is None
        assert not any(
            "fsdp" in item or "tensor_parallel" in item for item in result.applied
        )

    def test_single_device_with_no_transformations_says_so(self) -> None:
        result = parallelize(
            Toy(),
            ParallelDims(world_size=1),
            config=ParallelConfig(
                activation_checkpoint=ActivationCheckpointConfig(mode="none")
            ),
        )
        assert result.applied == ("single-device",)

    def test_fsdp_shards_across_dp_and_cp(self) -> None:
        dims = ParallelDims(world_size=32, context=4)
        total = sum(p.numel() for p in Toy().parameters())
        with fake_world(dims, device_type="cpu") as world:
            result = parallelize(Toy(), dims, mesh=world.require_mesh())
            local = sum(
                p.to_local().numel() if hasattr(p, "to_local") else p.numel()
                for p in result.model.parameters()
            )
        assert any("fsdp" in item for item in result.applied)
        # 8 shard ranks x 4 context ranks = 32-way sharding.
        assert local < total / 16

    def test_hsdp_keeps_replicate_distinct(self) -> None:
        dims = ParallelDims(world_size=32, dp_replicate=4, dp_shard=8)
        with fake_world(dims, device_type="cpu") as world:
            result = parallelize(Toy(), dims, mesh=world.require_mesh())
        assert any("hsdp" in item for item in result.applied)

    def test_tensor_parallel_refuses_to_guess_a_plan(self) -> None:
        # A wrong sharding plan trains silently to a wrong result, so a model
        # that has not declared one is a hard error.
        dims = ParallelDims(world_size=8, tensor=8)
        with (
            fake_world(dims, device_type="cpu") as world,
            pytest.raises(TypeError, match="tensor_parallel_plan"),
        ):
            parallelize(Toy(), dims, mesh=world.require_mesh())

    def test_unwrap_model_reaches_through_wrappers(self) -> None:
        model = Toy()
        apply_activation_checkpointing(
            model, ActivationCheckpointConfig(mode="selective_op")
        )
        assert isinstance(unwrap_model(model.blocks[0]), Block)


class TestPlans:
    def test_standard_block_plan_covers_every_projection(self) -> None:
        plan = standard_block_plan()
        for key in (
            "attention.q_proj",
            "attention.out_proj",
            "feed_forward.gate_proj",
            "feed_forward.down_proj",
        ):
            assert key in plan

    def test_sequence_parallel_adds_the_norm_regions(self) -> None:
        # Without it, TP saves weight memory but not activation memory — and for
        # video the activations are the problem.
        with_sp = standard_block_plan(sequence_parallel=True)
        without = standard_block_plan(sequence_parallel=False)
        assert "attention_norm" in with_sp
        assert "attention_norm" not in without


class TestPipeline:
    def test_split_points_account_for_embedding_and_head(self) -> None:
        points = balanced_split_points(32, 4, embedding_weight=0.5, head_weight=0.5)
        assert len(points) == 3
        assert points == sorted(points)
        assert 0 < points[0] < points[-1] < 32

    def test_refuses_more_stages_than_blocks(self) -> None:
        with pytest.raises(ValueError, match="cannot split"):
            balanced_split_points(3, 4)

    def test_bubble_shrinks_with_more_microbatches(self) -> None:
        config = PipelineConfig(microbatches=4)
        many = PipelineConfig(microbatches=64)
        assert many.bubble_fraction(8) < config.bubble_fraction(8)

    def test_interleaving_requires_an_interleaved_schedule(self) -> None:
        with pytest.raises(ValueError, match="single-stage"):
            PipelineConfig(schedule="1f1b", stages_per_rank=2)


class TestPrecision:
    def test_reduce_dtype_defaults_to_fp32(self) -> None:
        # bf16 reduction accumulates error through a log(world_size)-deep tree,
        # so the error grows with job size — exactly backwards.
        assert PrecisionConfig().reduce is torch.float32
        assert PrecisionConfig().param is torch.bfloat16

    def test_float16_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match="loss scaling"):
            PrecisionConfig(param_dtype="float16")

    def test_float8_without_torchao_is_an_explicit_error(self) -> None:
        from avgen.parallel import convert_to_float8

        config = PrecisionConfig(enable_float8=True)
        try:
            import torchao  # noqa: F401
        except ImportError:
            with pytest.raises(RuntimeError, match="torchao"):
                convert_to_float8(Toy(), config)

    def test_disabled_float8_converts_nothing(self) -> None:
        from avgen.parallel import convert_to_float8

        assert convert_to_float8(Toy(), PrecisionConfig()) == 0


class TestParallelConfig:
    def test_defaults_are_the_recommended_settings(self) -> None:
        config = ParallelConfig()
        assert config.sequence_parallel is True
        assert config.activation_checkpoint.mode == "selective_op"
        assert config.precision.reduce_dtype == "float32"


class TestContextParallelGradientSync:
    """A cp-only job must still reduce gradients.

    Each context-parallel rank computes the loss over its own slice of the
    sequence, so each holds a *partial* gradient for every shared parameter and
    the true gradient is their sum. When no data-parallel dimension exists there
    is nothing else to reduce over, so ``cp`` must serve as the sharding mesh
    itself. Missing this trains the model against something that is not the
    objective, and nothing in the loss curve says so.
    """

    def test_cp_only_still_shards(self) -> None:
        dims = ParallelDims(world_size=8, context=8)
        assert not dims.dp_shard_enabled
        with fake_world(dims, device_type="cpu") as world:
            result = parallelize(Toy(), dims, mesh=world.require_mesh())
        assert any(item == "fsdp(cp=8)" for item in result.applied)

    def test_cp_with_dp_folds_both_into_the_shard_group(self) -> None:
        dims = ParallelDims(world_size=8, dp_shard=4, context=2)
        with fake_world(dims, device_type="cpu") as world:
            result = parallelize(Toy(), dims, mesh=world.require_mesh())
        assert any("dp_shard=4, cp=2" in item for item in result.applied)
