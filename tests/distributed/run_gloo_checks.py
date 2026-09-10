"""Distributed checks that need collectives to actually move bytes.

Run under torchrun::

    torchrun --standalone --nproc-per-node 4 tests/distributed/run_gloo_checks.py

Everything in ``tests/test_distributed_stress.py`` runs against a
``FakeProcessGroup``, which is exact about shapes, sharding and collective
*patterns* — and says nothing about values, because fake collectives return
uninitialised memory. The four properties below are precisely the ones that
depend on a collective computing the right answer:

* a reduced gradient equals the mean of the inputs
* replicas stay bit-identical across many steps on different data
* a context-parallel loss equals the unsharded loss
* a checkpoint written by N ranks reloads correctly

This runs over **gloo on CPU**. That is not NCCL and is not a cluster, but the
thing under test is avgen's logic, not the transport: reduction correctness,
RNG discipline and resharding are backend-independent. What gloo cannot tell you
is anything about bandwidth, topology, or NCCL's own failure modes.

A script rather than a pytest module on purpose. Every check must execute on
every rank in the same order; pytest's collection and fixtures do not guarantee
that, and a rank that skips a check its peers are running deadlocks the job on
the next collective.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from avgen.core import (  # noqa: E402
    GridPatchifier,
    MediaBatch,
    RNGStreams,
    TrainState,
)
from avgen.data import SyntheticConfig, SyntheticSource  # noqa: E402
from avgen.models import VideoDiT, preset  # noqa: E402
from avgen.parallel import (  # noqa: E402
    ParallelDims,
    all_reduce_mean,
    data_mesh,
    parallelize,
    submesh,
)
from avgen.train import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
    MultiTaskConditioning,
    build_optimizer,
    build_timestep_sampler,
    train_step,
)

Failures = list[str]
CHECKS: dict[str, Callable[[], Failures]] = {}

# Built once in main(). Every call to init_device_mesh creates new sub-process
# groups; doing that per check exhausts gloo and the job stalls with no error
# and no traceback. Real training builds the mesh once, and so does this.
_MESH: dict[str, Any] = {}


def mesh_for(dims: ParallelDims) -> Any:
    """Return the process-wide mesh for these dims, building it once."""
    key = dims.describe()
    if key not in _MESH:
        _MESH[key] = dims.build_mesh("cpu")
    return _MESH[key]


def check(name: str) -> Callable[[Callable[[], Failures]], Callable[[], Failures]]:
    """Register a check under a name."""

    def wrap(fn: Callable[[], Failures]) -> Callable[[], Failures]:
        CHECKS[name] = fn
        return fn

    return wrap


def rank() -> int:
    """This rank's index."""
    return dist.get_rank()


def world() -> int:
    """Total ranks."""
    return dist.get_world_size()


def log(message: str) -> None:
    """Print from rank 0 only."""
    if rank() == 0:
        print(message, flush=True)


def build_model(seed: int = 0) -> nn.Module:
    """Build the tiny model with identical weights on every rank."""
    torch.manual_seed(seed)
    model = VideoDiT(preset("tiny"))
    model.init_weights()
    return model


def make_source(seed: int, samples: int = 16, *, frames: int = 8) -> SyntheticSource:
    """Build a deterministic synthetic corpus."""
    config = preset("tiny")
    return SyntheticSource(
        SyntheticConfig(
            seed=seed,
            num_samples=samples,
            video_channels=config.in_channels,
            frames=frames,
            height=8,
            width=8,
            audio_frames=0,
            text_tokens=8,
            text_width=config.text_width,
        )
    )


def make_objective() -> FlowMatchingObjective:
    """Build the standard flow-matching objective."""
    return FlowMatchingObjective(
        FlowMatchingConfig(),
        build_timestep_sampler("uniform"),
        MultiTaskConditioning(),
    )


def local_of(parameter: torch.Tensor) -> torch.Tensor:
    """Return a parameter's local shard, or the parameter itself."""
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def all_agree(tensor: torch.Tensor, *, tol: float = 0.0) -> bool:
    """Whether every rank holds the same tensor.

    Only valid for replicated tensors. It must never be handed an FSDP shard:
    ranks hold different-sized local shards of a small parameter, and
    ``all_gather`` with mismatched shapes does not error — it hangs. Use
    :func:`full_of` first.
    """
    parts = [torch.empty_like(tensor) for _ in range(world())]
    dist.all_gather(parts, tensor.contiguous())
    return all(torch.allclose(parts[0], other, atol=tol, rtol=tol) for other in parts)


def full_of(tensor: torch.Tensor) -> torch.Tensor:
    """Return the unsharded tensor, gathering a DTensor if necessary."""
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


@check("reduction_is_arithmetically_correct")
def reduction_is_arithmetically_correct() -> Failures:
    """The reduction must compute the actual mean, not merely complete.

    Under a fake process group this passes vacuously — the collective returns
    uninitialised memory and any assertion on its value is meaningless. It is
    the first thing worth checking once the bytes are real.
    """
    failures: Failures = []
    dims = ParallelDims(world_size=world())
    mesh = mesh_for(dims)
    value = torch.tensor([float(rank() + 1)])
    reduced = all_reduce_mean(value, data_mesh(mesh))
    expected = sum(range(1, world() + 1)) / world()
    if abs(float(reduced.item()) - expected) > 1e-6:
        failures.append(f"dp_cp mean is {float(reduced.item())}, expected {expected}")
    return failures


@check("gradients_are_reduced")
def gradients_are_reduced() -> Failures:
    """The gathered gradient must equal the mean of the per-rank gradients.

    Note what is *not* asserted: that every rank holds the same local gradient.
    Under FSDP each rank owns a different shard of the reduce-scattered result,
    so the shards are supposed to differ — comparing them directly is both wrong
    and, because the shards have different sizes for a small parameter, hangs
    rather than fails.

    The real property is that the reduction computed the right value. Each rank
    is fed different data; the gathered gradient must then match the average of
    what each rank would have produced on its own. If reduction is missing, the
    run silently becomes N independent models and no loss curve shows it.
    """
    failures: Failures = []
    dims = ParallelDims(world_size=world())
    mesh = mesh_for(dims)
    data_rank, _ = dims.data_coordinates(mesh)
    batch = next(iter(make_source(100 + data_rank)))
    objective, patchifier = make_objective(), GridPatchifier()

    # What this rank alone would compute, unsharded.
    solo = build_model()
    solo_out = objective(
        solo, batch, RNGStreams.for_rank(0, data_rank=data_rank), patchifier=patchifier
    )
    solo_out.loss.backward()
    solo_grads = {
        n: p.grad.clone() for n, p in solo.named_parameters() if p.grad is not None
    }

    # The mean across ranks of those solo gradients is the target.
    target: dict[str, torch.Tensor] = {}
    for name in sorted(solo_grads):
        stacked = solo_grads[name].clone()
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        target[name] = stacked / world()

    # Now the same step through the sharded path.
    parallel = parallelize(build_model(), dims, mesh=mesh)
    state = TrainState(
        model=parallel.model,
        optimizer=build_optimizer(parallel.model, lr=1e-3, weight_decay=0.0),
        rng=RNGStreams.for_rank(0, data_rank=data_rank),
    )
    train_step(state, batch, objective, patchifier=patchifier)

    for name, parameter in parallel.model.named_parameters():
        if parameter.grad is None:
            continue
        key = name.replace("_checkpoint_wrapped_module.", "")
        if key not in target:
            continue
        gathered = full_of(parameter.grad)
        # The sharded path computes in bfloat16 while the reference above is
        # fp32, so the comparison is relative and sized to bf16's precision
        # (eight mantissa bits, about 4e-3 relative). A missing reduction is
        # off by a factor, not by a few ulp, so this still catches it loudly.
        scale = max(float(target[key].abs().max()), 1e-6)
        delta = float((gathered - target[key]).abs().max()) / scale
        if delta > 2e-2:
            failures.append(
                f"{name}: reduced gradient differs from the mean of per-rank "
                f"gradients by {delta:.1%} relative — reduction is not "
                "computing the mean"
            )
            break
    return failures


@check("replicas_stay_identical")
def replicas_stay_identical() -> Failures:
    """After many steps on different data, every rank holds the same weights.

    A single missed reduction shows up here as drift that grows with step
    count — which is exactly how it presents in a real run, and why one step is
    not enough to catch it.
    """
    failures: Failures = []
    dims = ParallelDims(world_size=world())
    mesh = mesh_for(dims)
    data_rank, _ = dims.data_coordinates(mesh)
    parallel = parallelize(build_model(), dims, mesh=mesh)
    state = TrainState(
        model=parallel.model,
        optimizer=build_optimizer(parallel.model, lr=1e-2, weight_decay=0.0),
        rng=RNGStreams.for_rank(0, data_rank=data_rank),
    )
    objective, patchifier = make_objective(), GridPatchifier()
    iterator = iter(make_source(200 + data_rank))
    for _ in range(10):
        train_step(state, next(iterator), objective, patchifier=patchifier)
    for name, parameter in parallel.model.named_parameters():
        # full_of, not the local shard: shards legitimately differ under FSDP.
        if not all_agree(full_of(parameter).detach(), tol=1e-5):
            failures.append(f"weights drifted apart after 10 steps: {name}")
            break
    return failures


@check("context_parallel_loss_matches")
def context_parallel_loss_matches() -> Failures:
    """The reported loss must be global; the returned loss deliberately is not.

    This distinction is the whole design and is easy to test wrongly.

    ``ObjectiveOutput.loss`` under context parallelism is a *per-rank* quantity:
    a local numerator over the globally-reduced denominator, multiplied by the
    context degree. It is built that way on purpose. FSDP averages gradients
    over the shard group, and the context dimension *is* that group here, so the
    average of ``degree * num_r / den_global`` across ranks differentiates to
    exactly the global gradient. Asserting that this number equals the unsharded
    loss would be asserting against the design.

    ``video_loss`` is the reduced, reportable metric — that is what must equal
    the unsharded value, and what a training log should show.
    """
    failures: Failures = []
    if world() < 2:
        return failures
    dims = ParallelDims(world_size=world(), context=world())
    cp_mesh = submesh(mesh_for(dims), "cp")
    model = build_model(seed=5)
    # Size the clip so its token count divides by the context degree. avgen
    # refuses to pad implicitly — the padding would change the token count the
    # loss normalises by — so a real job sizes its buckets for the degree, and
    # so does this. With 2x2 patching an 8x8 latent gives 16 tokens per frame,
    # so 4 * world frames always divides.
    batch = next(iter(make_source(11, frames=4 * world())))
    objective, patchifier = make_objective(), GridPatchifier()

    whole = objective(model, batch, RNGStreams.from_seed(0), patchifier=patchifier)
    sharded = objective(
        model, batch, RNGStreams.from_seed(0), patchifier=patchifier, cp_mesh=cp_mesh
    )

    # The reported metric is global and must match.
    delta = abs(float(whole.video_loss) - float(sharded.video_loss))
    if delta > 1e-3 * max(1.0, abs(float(whole.video_loss))):
        failures.append(
            f"reported video_loss {float(sharded.video_loss):.6f} != unsharded "
            f"{float(whole.video_loss):.6f} (delta {delta:.2e})"
        )

    # And the identity the per-rank scaling exists to satisfy.
    #
    # Each rank returns degree * num_r / den_global. FSDP *averages* gradients
    # over the shard group — which is the context dimension here — so the
    # gradient that lands is (1/degree) * sum_r grad(degree * num_r / den) =
    # grad(sum_r num_r / den) = grad(global loss). Exactly right.
    #
    # The observable consequence is that the mean of the per-rank losses equals
    # the global loss. Drop the degree factor from the objective and this comes
    # out 1/degree too small, and the model quietly trains at 1/degree of the
    # intended learning rate.
    mean_loss = all_reduce_mean(sharded.loss.detach().clone(), cp_mesh)
    expected = float(whole.loss)
    if abs(float(mean_loss) - expected) > 1e-2 * max(1.0, abs(expected)):
        failures.append(
            f"mean of per-rank cp loss is {float(mean_loss):.6f}, expected the "
            f"global loss {expected:.6f} — the degree scaling is wrong, and the "
            "effective learning rate is off by a factor of the context degree"
        )

    # Token counts must also be global, or throughput accounting is wrong.
    if int(whole.valid_video_tokens) != int(sharded.valid_video_tokens):
        failures.append(
            f"valid_video_tokens {int(sharded.valid_video_tokens)} != "
            f"{int(whole.valid_video_tokens)} — the count was not reduced"
        )
    return failures


@check("checkpoint_roundtrip")
def checkpoint_roundtrip() -> Failures:
    """A sharded checkpoint must restore weights, optimizer state and progress."""
    from avgen.checkpoint import CheckpointManager

    failures: Failures = []
    payload: list[Any] = [
        tempfile.mkdtemp(prefix="avgen-gloo-") if rank() == 0 else None
    ]
    dist.broadcast_object_list(payload, src=0)
    directory = Path(str(payload[0])) / "ckpt"

    dims = ParallelDims(world_size=world())
    mesh = mesh_for(dims)
    parallel = parallelize(build_model(seed=3), dims, mesh=mesh)
    state = TrainState(
        model=parallel.model,
        optimizer=build_optimizer(parallel.model, lr=1e-3, weight_decay=0.0),
        rng=RNGStreams.for_rank(0, data_rank=0),
        step=17,
    )
    for parameter in parallel.model.parameters():
        shard = local_of(parameter)
        shard.grad = torch.randn_like(shard) * 0.01
    state.optimizer.step()

    before = {n: local_of(p).clone() for n, p in parallel.model.named_parameters()}
    manager = CheckpointManager(directory)
    manager.save(state.step, state, parallel=parallel)
    manager.wait()
    dist.barrier()

    with torch.no_grad():
        for parameter in parallel.model.parameters():
            local_of(parameter).add_(1.0)
    state.step = 0
    manager.load(state, parallel=parallel)

    if state.step != 17:
        failures.append(f"step restored as {state.step}, expected 17")
    for name, parameter in parallel.model.named_parameters():
        if not torch.allclose(local_of(parameter), before[name], atol=1e-6):
            failures.append(f"{name} not restored")
            break
    dist.barrier()
    if rank() == 0:
        shutil.rmtree(directory.parent, ignore_errors=True)
    return failures


@check("nonfinite_on_one_rank_does_not_desync")
def nonfinite_on_one_rank_does_not_desync() -> Failures:
    """A NaN on ONE rank must not desynchronise the others.

    The nastiest failure mode in the system. If one rank skips its optimizer
    step while its peers take theirs, the replicas diverge silently and every
    later all-reduce mixes two different models. If instead the skip decision
    needs a collective that only some ranks reach, the job deadlocks. The
    barrier below is where a deadlock would surface.
    """
    failures: Failures = []
    dims = ParallelDims(world_size=world())
    mesh = mesh_for(dims)
    data_rank, _ = dims.data_coordinates(mesh)
    parallel = parallelize(build_model(seed=9), dims, mesh=mesh)
    state = TrainState(
        model=parallel.model,
        optimizer=build_optimizer(parallel.model, lr=1e-2, weight_decay=0.0),
        rng=RNGStreams.for_rank(0, data_rank=data_rank),
    )
    batch = next(iter(make_source(300 + data_rank)))
    if rank() == world() - 1:
        batch = MediaBatch(
            video=batch.video * float("nan"),
            audio=batch.audio,
            text=batch.text,
            video_mask=batch.video_mask,
            audio_mask=batch.audio_mask,
            video_positions=batch.video_positions,
            audio_positions=batch.audio_positions,
            sample_ids=batch.sample_ids,
            spec=batch.spec,
            text_mask=batch.text_mask,
            targets=batch.targets,
        )
    state, _ = train_step(state, batch, make_objective(), patchifier=GridPatchifier())
    dist.barrier()

    if not all_agree(torch.tensor([float(state.step)])):
        failures.append("ranks disagree about the step count after a one-rank NaN")
    for name, parameter in parallel.model.named_parameters():
        gathered = full_of(parameter)
        if torch.isnan(gathered).any():
            failures.append(f"NaN leaked into {name}")
            break
        if not all_agree(gathered.detach(), tol=1e-5):
            failures.append(f"weights diverged after a one-rank NaN: {name}")
            break
    return failures


@check("data_shards_are_disjoint")
def data_shards_are_disjoint() -> Failures:
    """Data must partition across data ranks and replicate within a group."""
    failures: Failures = []
    dims = ParallelDims(world_size=world())
    mesh = mesh_for(dims)
    data_rank, data_world = dims.data_coordinates(mesh)

    total = 16
    mine = sorted(range(data_rank, total, data_world))
    payload: list[Any] = [None] * world()
    dist.all_gather_object(payload, (data_rank, mine))

    groups: dict[int, list[int]] = {}
    for group, indices in payload:  # type: ignore[misc]
        if group in groups and groups[group] != indices:
            failures.append(
                f"ranks sharing data_rank {group} disagree about their slice"
            )
        groups[group] = indices
    seen: set[int] = set()
    for group, indices in groups.items():
        overlap = seen & set(indices)
        if overlap:
            failures.append(
                f"data_rank {group} overlaps an earlier shard on {sorted(overlap)}"
            )
        seen |= set(indices)
    if seen != set(range(total)):
        failures.append(f"shards miss {sorted(set(range(total)) - seen)}")
    return failures


def main() -> int:
    """Run the checks and aggregate failures across ranks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    arguments = parser.parse_args()

    dist.init_process_group("gloo")
    torch.manual_seed(0)
    selected = arguments.only or list(CHECKS)

    log("=" * 74)
    log(f"avgen distributed checks — {world()} real ranks over gloo")
    log("=" * 74)

    total = 0
    for name in selected:
        dist.barrier()
        if os.environ.get("AVGEN_TRACE"):
            print(f"[rank {rank()}] -> {name}", flush=True)
        try:
            failures = CHECKS[name]()
        except Exception:
            failures = [f"raised: {traceback.format_exc(limit=5)}"]
        payload: list[Any] = [None] * world()
        dist.all_gather_object(payload, failures)
        merged = [
            f"[rank {i}] {item}"
            for i, part in enumerate(payload)
            for item in part or []
        ]
        if merged:
            total += len(merged)
            log(f"  FAIL  {name}")
            for item in merged[:4]:
                log(f"          {item}")
        else:
            log(f"  ok    {name}")

    log("-" * 74)
    log("ALL CHECKS PASSED" if total == 0 else f"{total} FAILURE(S)")
    dist.destroy_process_group()
    return 0 if total == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
