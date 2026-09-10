"""Tests for the plan simulator.

The simulator's job is to be *directionally* right and *structurally* exact: it
must get shapes, sharding, and collective patterns exactly right, and memory and
time within a useful margin. These tests assert the exact parts strictly and the
estimated parts as monotonic relationships — an estimate that moves the wrong
way when you turn a knob is broken regardless of its absolute error.
"""

from __future__ import annotations

import json

import pytest

from avgen.parallel import ActivationCheckpointConfig, ParallelDims
from avgen.simulate import (
    B200,
    H100_SXM,
    NVLINK4,
    Interconnect,
    ModelShape,
    SearchSpace,
    calibrate_from_busbw,
    estimate_compute,
    estimate_memory,
    estimate_step_communication,
    render_plan_table,
    search_parallel_plan,
    simulate_config,
    suggest_activation_policy,
    transformer_flops,
)

SHAPE = ModelShape(
    parameters=2_000_000_000,
    depth=32,
    width=2560,
    sequence_length=65_536,
    micro_batch_size=1,
    num_heads=20,
    text_tokens=256,
)


class TestMemory:
    def test_sharding_reduces_parameter_memory_proportionally(self) -> None:
        one = estimate_memory(SHAPE, ParallelDims(world_size=1))
        many = estimate_memory(SHAPE, ParallelDims(world_size=64))
        assert many.parameter_bytes == pytest.approx(one.parameter_bytes / 64)
        assert many.optimizer_bytes == pytest.approx(one.optimizer_bytes / 64)

    def test_context_parallel_reduces_activation_memory(self) -> None:
        # The whole point of CP: nothing else touches the activation term.
        without = estimate_memory(SHAPE, ParallelDims(world_size=64))
        with_cp = estimate_memory(SHAPE, ParallelDims(world_size=64, context=8))
        assert with_cp.activation_bytes < without.activation_bytes / 4

    def test_activation_checkpointing_trades_memory_for_compute(self) -> None:
        dims = ParallelDims(world_size=64, context=4)
        none = estimate_memory(SHAPE, dims)
        selective = estimate_memory(
            SHAPE,
            dims,
            activation_checkpoint=ActivationCheckpointConfig("selective_op"),
        )
        full = estimate_memory(
            SHAPE, dims, activation_checkpoint=ActivationCheckpointConfig("full")
        )
        assert full.activation_bytes < selective.activation_bytes
        assert selective.activation_bytes < none.activation_bytes

    def test_adamw_state_is_twelve_bytes_per_parameter(self) -> None:
        # Not eight: an fp32 master copy plus two fp32 moments. Getting this
        # wrong understates optimizer memory by a third.
        estimate = estimate_memory(SHAPE, ParallelDims(world_size=1))
        assert estimate.optimizer_bytes == pytest.approx(SHAPE.parameters * 12)

    def test_unknown_optimizer_is_a_clear_error(self) -> None:
        with pytest.raises(KeyError, match="unknown optimizer"):
            estimate_memory(SHAPE, ParallelDims(world_size=1), optimizer="nonexistent")

    def test_fits_in_keeps_headroom(self) -> None:
        estimate = estimate_memory(SHAPE, ParallelDims(world_size=512, context=8))
        assert estimate.fits_in(80.0, headroom=0.10)
        assert not estimate.fits_in(estimate.total_gib, headroom=0.50)

    def test_dominant_term_names_what_to_attack(self) -> None:
        estimate = estimate_memory(SHAPE, ParallelDims(world_size=8))
        assert estimate.dominant_term() in {
            "activation_bytes",
            "optimizer_bytes",
            "parameter_bytes",
            "gradient_bytes",
            "gather_bytes",
            "workspace_bytes",
        }


class TestCompute:
    def test_attention_dominates_at_long_sequence(self) -> None:
        # The reason the language-model FLOPs shortcut is wrong for video.
        short = transformer_flops(ModelShape(2e9, 32, 2560, 2_048, num_heads=20))
        long = transformer_flops(SHAPE)
        assert short["attention_fraction"] < 0.2
        assert long["attention_fraction"] > 0.6

    def test_flops_scale_quadratically_in_sequence(self) -> None:
        base = transformer_flops(ModelShape(2e9, 32, 2560, 8_192, num_heads=20))
        double = transformer_flops(ModelShape(2e9, 32, 2560, 16_384, num_heads=20))
        ratio = double["attention"] / base["attention"]
        assert ratio == pytest.approx(4.0)

    def test_recompute_overhead_is_reported_separately(self) -> None:
        dims = ParallelDims(world_size=64)
        none = estimate_compute(SHAPE, dims)
        full = estimate_compute(
            SHAPE, dims, activation_checkpoint=ActivationCheckpointConfig("full")
        )
        assert none.recompute_overhead == 0.0
        assert full.recompute_overhead == pytest.approx(1 / 3)
        # Model FLOPs (the MFU numerator) must not include recomputation.
        assert full.model_flops == pytest.approx(none.model_flops)

    def test_faster_device_predicts_less_time(self) -> None:
        dims = ParallelDims(world_size=64)
        assert (
            estimate_compute(SHAPE, dims, accelerator=B200).seconds
            < estimate_compute(SHAPE, dims, accelerator=H100_SXM).seconds
        )

    def test_suggest_policy_picks_the_cheapest_that_fits(self) -> None:
        roomy = suggest_activation_policy(
            SHAPE, ParallelDims(world_size=512, context=8), accelerator=H100_SXM
        )
        tight = suggest_activation_policy(
            SHAPE, ParallelDims(world_size=8), accelerator=H100_SXM
        )
        assert roomy.mode == "none"
        assert tight.mode != "none"


class TestCommunication:
    def test_context_parallel_adds_ring_traffic(self) -> None:
        without = estimate_step_communication(SHAPE, ParallelDims(world_size=64))
        with_cp = estimate_step_communication(
            SHAPE, ParallelDims(world_size=64, context=8)
        )
        labels = with_cp.by_label()
        assert "context_parallel.ring_kv" in labels
        assert "context_parallel.ring_kv" not in without.by_label()

    def test_ring_formula_matches_the_textbook_factor(self) -> None:
        # all-reduce moves 2S(n-1)/n; reduce-scatter moves S(n-1)/n.
        estimate = estimate_step_communication(
            SHAPE, ParallelDims(world_size=16, dp_replicate=4, dp_shard=4)
        )
        allreduce = next(
            item for item in estimate.collectives if item.kind == "all_reduce"
        )
        expected = (
            2.0 * allreduce.payload_bytes * (allreduce.ranks - 1) / allreduce.ranks
        )
        assert allreduce.wire_bytes == pytest.approx(expected)

    def test_overlap_hides_communication_behind_compute(self) -> None:
        dims = ParallelDims(world_size=64, context=4)
        compute = estimate_compute(SHAPE, dims)
        overlapped = estimate_step_communication(
            SHAPE, dims, compute_seconds=compute.seconds, overlap_efficiency=0.9
        )
        exposed = estimate_step_communication(
            SHAPE, dims, compute_seconds=compute.seconds, overlap_efficiency=0.0
        )
        assert overlapped.exposed_seconds < exposed.exposed_seconds
        assert overlapped.scaling_efficiency > exposed.scaling_efficiency

    def test_calibration_from_nccl_tests(self) -> None:
        # Bus bandwidth already folds in the ring factor, so dividing by peak
        # gives the efficiency term directly.
        tuned = calibrate_from_busbw(
            "NVLink 4", measured_busbw_gbps=372.0, peak_gbps=450.0
        )
        assert tuned.efficiency == pytest.approx(372.0 / 450.0)
        assert tuned.achievable_bytes_per_second < NVLINK4.peak_gbps * 1e9

    def test_slow_fabric_costs_more(self) -> None:
        slow = Interconnect("slow", peak_gbps=5.0, efficiency=0.5)
        dims = ParallelDims(world_size=64, context=4)
        fast_estimate = estimate_step_communication(SHAPE, dims)
        slow_estimate = estimate_step_communication(
            SHAPE, dims, intra_node=slow, inter_node=slow
        )
        assert slow_estimate.total_seconds > fast_estimate.total_seconds


class TestPlanSearch:
    def test_returns_only_configurations_that_fit(self) -> None:
        plans = search_parallel_plan(SHAPE, SearchSpace(world_size=512))
        assert plans
        for plan in plans:
            assert plan.memory.fits_in(H100_SXM.memory_gib)
            assert plan.dims.world_size == 512

    def test_never_puts_tensor_parallel_across_nodes(self) -> None:
        plans = search_parallel_plan(
            SHAPE, SearchSpace(world_size=512, gpus_per_node=4, max_tensor=8)
        )
        assert all(plan.dims.tensor <= 4 for plan in plans)

    def test_respects_a_pinned_global_batch(self) -> None:
        plans = search_parallel_plan(
            SHAPE, SearchSpace(world_size=512, global_batch_size=512)
        )
        for plan in plans:
            product = (
                plan.dims.dp_size * plan.micro_batch_size * plan.gradient_accumulation
            )
            assert product == 512

    def test_context_parallel_must_divide_the_sequence(self) -> None:
        odd = ModelShape(2e9, 32, 2560, 65_537, num_heads=20)
        plans = search_parallel_plan(odd, SearchSpace(world_size=512, max_context=16))
        assert all(plan.dims.sequence_shard_size == 1 for plan in plans)

    def test_impossible_configuration_returns_nothing_with_advice(self) -> None:
        huge = ModelShape(400e9, 128, 16384, 262_144, num_heads=128)
        plans = search_parallel_plan(huge, SearchSpace(world_size=8))
        assert plans == []
        assert "No configuration fits" in render_plan_table(plans, H100_SXM)

    def test_ties_break_on_memory_headroom(self) -> None:
        # With a pinned global batch, throughput is near-identical across plans
        # by construction; the tiebreak must then prefer more headroom.
        plans = search_parallel_plan(
            SHAPE, SearchSpace(world_size=512, global_batch_size=512), top_k=5
        )
        assert plans[0].memory.total_gib <= min(p.memory.total_gib for p in plans)

    def test_table_renders(self) -> None:
        table = render_plan_table(
            search_parallel_plan(SHAPE, SearchSpace(world_size=512), top_k=3), H100_SXM
        )
        assert "mfu" in table
        assert "bottleneck" in table


class TestReport:
    def test_round_trips_through_json(self, tmp_path) -> None:
        report = simulate_config(SHAPE, ParallelDims(world_size=512, context=8))
        path = report.save(tmp_path / "sim.json")
        restored = json.loads(path.read_text())
        assert restored["plan"] == report.dims.describe()
        assert restored["fits"] is report.fits

    def test_renders_a_readable_summary(self) -> None:
        text = simulate_config(SHAPE, ParallelDims(world_size=512, context=8)).render()
        for section in ("MEMORY", "COMPUTE", "COMMUNICATION", "THROUGHPUT"):
            assert section in text

    def test_advises_on_long_sequence_attention(self) -> None:
        report = simulate_config(SHAPE, ParallelDims(world_size=512, context=8))
        assert any("attention is" in note for note in report.notes)

    def test_warns_when_tensor_parallel_spans_nodes(self) -> None:
        # tp=16 cannot fit in an 8-GPU NVLink domain.
        report = simulate_config(SHAPE, ParallelDims(world_size=512, tensor=16))
        assert any("spans nodes" in note for note in report.notes)

    def test_regression_gate_catches_a_memory_increase(self) -> None:
        # Halving context parallelism roughly doubles activation memory while
        # still fitting, which is the subtle regression the gate exists for.
        baseline = simulate_config(
            SHAPE, ParallelDims(world_size=512, context=8)
        ).to_dict()
        worse = simulate_config(SHAPE, ParallelDims(world_size=512, context=4))
        assert worse.fits
        with pytest.raises(AssertionError, match="per-rank memory grew"):
            worse.assert_no_regression(baseline)

    def test_regression_gate_reports_a_plan_that_stopped_fitting(self) -> None:
        baseline = simulate_config(
            SHAPE, ParallelDims(world_size=512, context=8)
        ).to_dict()
        broken = simulate_config(SHAPE, ParallelDims(world_size=512, context=2))
        assert not broken.fits
        with pytest.raises(AssertionError, match="no longer fits"):
            broken.assert_no_regression(baseline)

    def test_regression_gate_passes_an_unchanged_plan(self) -> None:
        dims = ParallelDims(world_size=512, context=8)
        baseline = simulate_config(SHAPE, dims).to_dict()
        simulate_config(SHAPE, dims).assert_no_regression(baseline)

    def test_days_for_tokens_is_a_budget_number(self) -> None:
        report = simulate_config(SHAPE, ParallelDims(world_size=512, context=8))
        assert report.days_for_tokens(1e12) > 0
        assert report.days_for_tokens(1e12) < report.days_for_tokens(1e13)
