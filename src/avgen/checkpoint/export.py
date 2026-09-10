"""Consolidated release artifacts: the checkpoint you give to somebody else.

A training checkpoint and a released model are different objects with different
requirements, and conflating them is why so many open weight releases ship a
directory nobody outside the original lab can load.

============  ==================================  ==============================
              Training checkpoint (DCP)            Release artifact (safetensors)
============  ==================================  ==============================
Contents      Weights + optimizer + RNG +          Weights only, consolidated.
              schedule + data cursor.
Layout        Sharded, one file per rank.          Full tensors, sharded by size.
Read by       The same framework, any rank         Anything: ``safetensors``,
              count.                               ``transformers``, ``diffusers``.
Mutability    Overwritten every few hundred        Immutable. It is a citation.
              steps.
Cost          Cheap: no gathering, no copy.        Expensive: all-gathers every
                                                   sharded parameter onto rank 0.
============  ==================================  ==============================

The expensive part is unavoidable. Under FSDP2 or tensor parallelism every
parameter is a :class:`~torch.distributed.tensor.DTensor` holding one shard, and
a consumer that has never heard of ``DeviceMesh`` needs the whole tensor.
``get_model_state_dict`` with ``full_state_dict=True`` issues the all-gathers;
pairing it with ``cpu_offload=True`` means only rank 0 materialises the result,
which is what keeps a 30B export from OOMing every rank at once.

Sharding the output by *bytes* rather than by tensor count is the other detail
that matters for distribution: many filesystems, object stores, and download
CDNs behave badly past a few gigabytes per object, so the convention — a set of
``model-000xx-of-000NN.safetensors`` files plus a ``model.safetensors.index.json``
weight map — exists to keep every object small enough to retry cheaply.

``safetensors`` itself is chosen over ``torch.save`` for one reason: loading a
pickle executes arbitrary code from the file. A weights format that can run a
shell command is not a format you can accept from a stranger, and a release
artifact is by definition consumed by strangers.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

__all__ = [
    "DEFAULT_MAX_SHARD_BYTES",
    "INDEX_FILENAME",
    "WEIGHTS_FILENAME",
    "convert",
    "export_huggingface",
    "export_safetensors",
    "import_safetensors",
]

_LOG = logging.getLogger("avgen.checkpoint")

#: The de facto standard names. Deviating from them costs nothing technically
#: and everything practically: every downstream loader globs for exactly these.
WEIGHTS_FILENAME = "model.safetensors"
INDEX_FILENAME = "model.safetensors.index.json"

#: 5 GiB per file. Large enough that a 30B bf16 model is a dozen files, small
#: enough that a failed download is a cheap retry rather than an hour lost.
DEFAULT_MAX_SHARD_BYTES = 5 * 1024**3


def _require_safetensors() -> Any:
    """Import ``safetensors.torch`` lazily with an actionable error.

    Returns:
        The ``safetensors.torch`` module.

    Raises:
        RuntimeError: If safetensors is not installed.
    """
    try:
        import safetensors.torch as safetensors_torch
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "safetensors is required for export/import; install 'avgen'"
        ) from error
    return safetensors_torch


def _is_main_rank() -> bool:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()) == 0
    return True


def _shard_name(index: int, total: int) -> str:
    """Return the conventional shard filename for one of ``total`` shards."""
    if total == 1:
        return WEIGHTS_FILENAME
    return f"model-{index + 1:05d}-of-{total:05d}.safetensors"


def _plan_shards(
    tensors: Mapping[str, torch.Tensor], max_shard_bytes: int
) -> list[list[str]]:
    """Group parameter names into byte-bounded shards, preserving order.

    Order is preserved rather than optimised (a bin-packing solver would pack
    tighter) because module definition order keeps a block's parameters in the
    same file, so a consumer streaming layer by layer reads each file once
    instead of seeking across all of them.
    """
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, tensor in tensors.items():
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current or not shards:
        shards.append(current)
    return shards


def _materialise(
    tensors: Mapping[str, torch.Tensor], dtype: torch.dtype | None
) -> dict[str, torch.Tensor]:
    """Make every tensor safetensors-writable: contiguous, unshared, on CPU.

    safetensors refuses tensors that alias one storage, because the format has
    no way to express aliasing and writing both copies would silently double the
    file size. Tied embeddings and weight-shared heads hit this constantly, so
    duplicates are cloned rather than reported as an error the caller cannot act
    on. A non-contiguous tensor (a transposed view, a slice) is cloned for the
    same reason: the format stores a flat buffer.
    """
    seen: dict[int, str] = {}
    output: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        value = tensor.detach()
        if value.device.type != "cpu":
            value = value.to("cpu")
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype)
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen or not value.is_contiguous():
            value = value.clone().contiguous()
            pointer = value.untyped_storage().data_ptr()
        seen[pointer] = name
        output[name] = value
    return output


def export_safetensors(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    dtype: torch.dtype | None = None,
    metadata: Mapping[str, str] | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> Path:
    """Write a consolidated, framework-agnostic weights directory.

    **Every rank must call this.** Gathering a sharded parameter is a
    collective; ranks other than 0 contribute their shards and then write
    nothing. Guarding the call with ``if rank == 0`` hangs the job.

    Args:
        path: Output directory, created if needed.
        model: The (possibly sharded) model to export.
        dtype: Cast floating-point tensors to this dtype. ``bfloat16`` halves
            the artifact for a model that will be run in bf16 anyway; leave it
            ``None`` to preserve exactly what was trained, which is the right
            default for anything that will be fine-tuned further.
        metadata: Extra string-to-string header fields — the run id, the config
            hash, the license. Written into every shard's header so a file
            separated from its directory can still be identified. Keys are
            strings only; that is a format restriction, not a choice.
        max_shard_bytes: Soft byte cap per file. A single tensor larger than
            this still occupies one file, since a tensor is never split.

    Returns:
        The output directory.

    Raises:
        ValueError: If ``max_shard_bytes`` is not positive.
    """
    if max_shard_bytes <= 0:
        raise ValueError(f"max_shard_bytes must be positive; got {max_shard_bytes}")
    safetensors_torch = _require_safetensors()
    directory = Path(path)

    # full_state_dict gathers every DTensor; cpu_offload keeps the result on
    # rank 0 only, so peak host memory is one model rather than world_size.
    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if not _is_main_rank():
        return directory

    tensors = _materialise(
        {
            name: value
            for name, value in state.items()
            if isinstance(value, torch.Tensor)
        },
        dtype,
    )
    directory.mkdir(parents=True, exist_ok=True)
    groups = _plan_shards(tensors, max_shard_bytes)

    header: dict[str, str] = {"format": "pt"}
    if metadata:
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "safetensors metadata must be str -> str; got "
                    f"{type(key).__name__} -> {type(value).__name__}"
                )
            header[key] = value

    weight_map: dict[str, str] = {}
    total_bytes = 0
    for index, names in enumerate(groups):
        filename = _shard_name(index, len(groups))
        payload = {name: tensors[name] for name in names}
        safetensors_torch.save_file(payload, directory / filename, metadata=header)
        for name in names:
            weight_map[name] = filename
            total_bytes += tensors[name].numel() * tensors[name].element_size()

    # The index is written last: its presence means the shard set is complete,
    # the same contract the training checkpoint's marker provides.
    index_payload = {
        "metadata": {"total_size": total_bytes, **header},
        "weight_map": weight_map,
    }
    (directory / INDEX_FILENAME).write_text(
        json.dumps(index_payload, indent=2, sort_keys=True)
    )
    _LOG.info(
        "exported %d tensors (%.2f GiB) across %d shard(s) -> %s",
        len(weight_map),
        total_bytes / 1024**3,
        len(groups),
        directory,
    )
    return directory


def _read_safetensors_dir(path: Path) -> dict[str, torch.Tensor]:
    """Load every shard of a safetensors directory (or a single file)."""
    safetensors_torch = _require_safetensors()
    if path.is_file():
        loaded: dict[str, torch.Tensor] = safetensors_torch.load_file(path)
        return loaded
    index = path / INDEX_FILENAME
    if index.is_file():
        payload = json.loads(index.read_text())
        files = sorted(set(payload["weight_map"].values()))
    else:
        # No index: accept a plain directory of shards so a hand-assembled
        # release still loads.
        files = sorted(child.name for child in path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors shards found under {path}")
    tensors: dict[str, torch.Tensor] = {}
    for filename in files:
        tensors.update(safetensors_torch.load_file(path / filename))
    return tensors


def import_safetensors(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    strict: bool = True,
) -> nn.Module:
    """Load a consolidated safetensors directory into a live model.

    Every rank reads the full artifact and ``set_model_state_dict`` scatters it
    into whatever placement the model now has, so a checkpoint exported from a
    512-rank run loads into a single GPU and vice versa. The rejected
    alternative — read on rank 0 and broadcast — halves the read bandwidth
    requirement but serialises the load behind one rank's filesystem; with a
    page cache shared across a node, every rank reading is usually faster.

    Args:
        path: Directory (or single ``.safetensors`` file) to read.
        model: The model to fill, in place.
        strict: Whether missing or unexpected parameter names are an error.
            Turn it off for a deliberate partial load, such as attaching a
            pretrained backbone to a new head.

    Returns:
        The same model, for chaining.

    Raises:
        FileNotFoundError: If no shards are found.
    """
    tensors = _read_safetensors_dir(Path(path))
    set_model_state_dict(
        model,
        model_state_dict=tensors,
        options=StateDictOptions(full_state_dict=True, strict=strict),
    )
    _LOG.info("imported %d tensors from %s", len(tensors), path)
    return model


def export_huggingface(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    dtype: torch.dtype | None = None,
    thread_count: int = 1,
    save_distributed: bool = False,
) -> Path:
    """Write weights through DCP's Hugging Face storage writer.

    This is the *sharded* path to the same on-disk format
    :func:`export_safetensors` produces. The difference is where the
    consolidation happens: here every rank writes its own shard directly in
    safetensors layout and DCP optionally consolidates afterwards, so rank 0
    never has to hold the whole model. For a model that fits comfortably in host
    memory, :func:`export_safetensors` is simpler and produces a tidier
    directory; past roughly 30B parameters the gather is the thing that fails,
    and this path is the one that works.

    Args:
        path: Output directory.
        model: The model to export.
        dtype: Optional floating-point cast, applied before writing.
        thread_count: Writer threads per rank.
        save_distributed: Whether each rank writes its own shard. ``False``
            gathers to rank 0 first, which matches the single-file layout most
            consumers expect; ``True`` is the memory-safe path for very large
            models and requires the consolidation step to run afterwards.

    Returns:
        The output directory.
    """
    directory = Path(path)
    options = StateDictOptions(
        full_state_dict=not save_distributed,
        cpu_offload=not save_distributed,
    )
    state = get_model_state_dict(model, options=options)
    if dtype is not None:
        state = {
            name: (
                value.to(dtype)
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value
            )
            for name, value in state.items()
        }
    if not save_distributed and not _is_main_rank():
        return directory
    if _is_main_rank():
        directory.mkdir(parents=True, exist_ok=True)
    writer = dcp.HuggingFaceStorageWriter(
        path=str(directory),
        thread_count=thread_count,
        save_distributed=save_distributed,
        # Consolidation turns per-rank shards into the conventional
        # model-000xx-of-000NN layout; without it the directory is readable only
        # by DCP, which defeats the purpose of exporting at all.
        enable_consolidation=save_distributed,
    )
    dcp.save(state, storage_writer=writer, no_dist=not save_distributed)
    _LOG.info("exported hugging-face format -> %s", directory)
    return directory


def _read_dcp_flat(source: Path) -> dict[str, Any]:
    """Read a whole DCP checkpoint into plain tensors, with no model to guide it.

    Uses DCP's empty-state-dict planner, which reads the metadata to discover
    what is in the checkpoint instead of being told. That is what makes offline
    conversion possible on a machine that cannot even construct the model.
    """
    from torch.distributed.checkpoint.format_utils import (
        _EmptyStateDictLoadPlanner,
        _load_state_dict,
    )

    state: dict[str, Any] = {}
    _load_state_dict(
        state,
        storage_reader=dcp.FileSystemReader(source),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state


def _extract_model_tensors(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Pull the model sub-tree out of a training checkpoint's state dict."""
    from avgen.checkpoint.stateful import MODEL_OPTIMIZER_KEY

    node: Any = state
    for key in (MODEL_OPTIMIZER_KEY, "model"):
        if not isinstance(node, Mapping) or key not in node:
            raise KeyError(
                f"checkpoint has no {MODEL_OPTIMIZER_KEY}/model sub-tree; "
                f"top-level keys are {sorted(state)}"
            )
        node = node[key]
    return {
        name: value for name, value in node.items() if isinstance(value, torch.Tensor)
    }


def convert(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    to: str = "safetensors",
    dtype: torch.dtype | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> Path:
    """Convert a checkpoint between formats offline, without building the model.

    This exists because the machine that can construct a 30B video DiT and the
    machine you happen to be sitting at are rarely the same one. Both directions
    read and write on a single process, so it runs on a login node.

    Args:
        src: Source checkpoint. A DCP directory when ``to="safetensors"``, a
            safetensors directory or file when ``to="dcp"``.
        dst: Destination directory.
        to: Target format, ``"safetensors"`` or ``"dcp"``.
        dtype: Optional floating-point cast.
        max_shard_bytes: Byte cap per safetensors shard.

    Returns:
        The destination directory.

    Raises:
        ValueError: If ``to`` is not a recognised format.
        KeyError: If a DCP source has no model sub-tree.
    """
    source = Path(src)
    destination = Path(dst)
    if to == "safetensors":
        safetensors_torch = _require_safetensors()
        tensors = _materialise(_extract_model_tensors(_read_dcp_flat(source)), dtype)
        destination.mkdir(parents=True, exist_ok=True)
        groups = _plan_shards(tensors, max_shard_bytes)
        weight_map: dict[str, str] = {}
        total_bytes = 0
        for index, names in enumerate(groups):
            filename = _shard_name(index, len(groups))
            safetensors_torch.save_file(
                {name: tensors[name] for name in names},
                destination / filename,
                metadata={"format": "pt"},
            )
            for name in names:
                weight_map[name] = filename
                total_bytes += tensors[name].numel() * tensors[name].element_size()
        (destination / INDEX_FILENAME).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_bytes, "format": "pt"},
                    "weight_map": weight_map,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif to == "dcp":
        tensors = _materialise(_read_safetensors_dir(source), dtype)
        destination.mkdir(parents=True, exist_ok=True)
        # Wrapped in the training layout so the result is loadable by
        # CheckpointManager as a warm start; the optimizer half is absent, which
        # a non-strict load tolerates.
        from avgen.checkpoint.stateful import MODEL_OPTIMIZER_KEY

        dcp.save(
            {MODEL_OPTIMIZER_KEY: {"model": tensors}},
            storage_writer=dcp.FileSystemWriter(destination, overwrite=True),
            no_dist=True,
        )
    else:
        raise ValueError(f"to must be 'safetensors' or 'dcp'; got {to!r}")
    _LOG.info("converted %s -> %s (%s)", source, destination, to)
    return destination
