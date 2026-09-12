"""``avgen info`` — what is installed, what imports, what hardware is visible.

The command exists because the first question about any failure on a cluster is
"is this a bug or is my environment wrong", and answering it usually costs
several minutes of interactive Python. It reports four things:

* **Versions** — Python, torch, CUDA, and avgen itself.
* **Devices** — count, name, compute capability, memory, and bf16/fp8 support.
  Compute capability is the one that matters: an fp8 recipe silently falls back
  on anything below 8.9, and a plan built assuming fp8 throughput is then wrong
  by a factor.
* **Optional extras** — which are importable, and the exact install command for
  the ones that are not.
* **avgen's own subsystems** — which import cleanly. On a partially installed
  tree, or a source checkout being developed by several people at once, this is
  the line that says which half is broken.

Nothing here imports torch at module scope and every probe is wrapped, because
a diagnostic command that fails on a broken environment is useless precisely
when it is needed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import sys
from typing import Any

from avgen.cli._common import emit, rule

__all__ = ["add_parser", "run"]

#: Optional extras, mapped to the modules that prove them present and the
#: extra that installs them.
_EXTRAS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("text", "frozen text towers (T5/Gemma/Qwen)", ("transformers", "sentencepiece")),
    ("codecs", "pretrained video/audio VAEs", ("diffusers",)),
    ("data", "media decode and columnar shards", ("pyarrow", "av", "torchaudio")),
    ("tracking", "experiment tracking", ("tensorboard",)),
    ("quant", "fp8 / low-precision training", ("torchao",)),
    ("fault-tolerance", "per-step recovery", ("torchft",)),
)

#: avgen's own subsystems, in dependency order so the first failure is usually
#: the root cause rather than a symptom.
_SUBSYSTEMS: tuple[str, ...] = (
    "avgen.core",
    "avgen.parallel",
    "avgen.simulate.plan",
    "avgen.config",
    "avgen.eval",
    "avgen.models",
    "avgen.data",
    "avgen.train",
    "avgen.checkpoint",
    "avgen.telemetry",
    "avgen.infer",
    "avgen.codecs",
    "avgen.finetune",
    "avgen.rl",
)


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``info`` subcommand.

    Args:
        subparsers: The parent's subparser action.

    Returns:
        The configured parser.
    """
    parser = subparsers.add_parser(
        "info",
        help="report versions, devices, optional extras, and subsystem health",
        description=(
            "Print everything needed to answer 'is this a bug or is my "
            "environment wrong'. Safe to run on a broken install: every probe "
            "is wrapped and reports the failure rather than raising it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="also list every evaluation metric and its availability",
    )
    parser.set_defaults(handler=run)
    return parser


def _module_version(name: str) -> str | None:
    """Return an importable module's version, or ``None`` when absent."""
    if importlib.util.find_spec(name.split(".")[0]) is None:
        return None
    try:
        module = importlib.import_module(name)
    except Exception as error:
        return f"present but not importable: {type(error).__name__}"
    return str(getattr(module, "__version__", "unknown version"))


def _avgen_version() -> str:
    """Return the installed avgen version, or a source-tree marker."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("avgen")
    except PackageNotFoundError:
        return "not installed (running from a source tree)"


def _render_devices() -> list[str]:
    """Probe torch and CUDA without ever raising."""
    lines: list[str] = []
    try:
        import torch
    except Exception as error:
        return [f"  torch          NOT IMPORTABLE: {error}"]

    lines.append(f"  torch          {torch.__version__}")
    built = torch.version.cuda or "cpu-only build"
    lines.append(f"  cuda (built)   {built}")
    try:
        available = torch.cuda.is_available()
    except Exception as error:
        lines.append(f"  cuda (runtime) probe failed: {error}")
        return lines
    if not available:
        lines.append("  cuda (runtime) no device visible — CPU only")
        lines.append(
            "                 'avgen plan' and 'avgen simulate' still work; "
            "they need no GPU"
        )
        return lines

    count = torch.cuda.device_count()
    lines.append(f"  cuda (runtime) {count} device(s)")
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        capability = f"{properties.major}.{properties.minor}"
        memory = properties.total_memory / 1024**3
        # sm_89 (Ada) and sm_90 (Hopper) are where fp8 tensor cores appear;
        # below that a float8 recipe silently falls back and every fp8-based
        # throughput projection is wrong by roughly a factor of two.
        fp8 = "yes" if (properties.major, properties.minor) >= (8, 9) else "no"
        lines.append(
            f"    [{index}] {properties.name} · sm_{capability.replace('.', '')} · "
            f"{memory:.0f} GiB · fp8 tensor cores: {fp8}"
        )
    return lines


def run(arguments: argparse.Namespace) -> int:
    """Print the environment report.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        ``0`` always. A diagnostic that fails is not a diagnostic; problems are
        reported in the output, not in the exit code.
    """
    emit(rule("avgen environment", character="="))
    emit()
    emit(rule("VERSIONS"))
    emit(f"  avgen          {_avgen_version()}")
    emit(f"  python         {platform.python_version()} ({sys.executable})")
    emit(f"  platform       {platform.platform()}")
    for line in _render_devices():
        emit(line)
    for name in ("numpy", "yaml", "safetensors"):
        emit(f"  {name:<14} {_module_version(name) or 'MISSING (core dependency)'}")
    emit()

    emit(rule("OPTIONAL EXTRAS"))
    for extra, purpose, modules in _EXTRAS:
        found = {name: _module_version(name) for name in modules}
        missing = [name for name, value in found.items() if value is None]
        if missing:
            status = f"missing {', '.join(missing)}"
            hint = f"pip install 'avgen[{extra}]'"
        else:
            status = ", ".join(f"{name} {value}" for name, value in found.items())
            hint = ""
        emit(f"  {extra:<16} {purpose}")
        emit(f"  {'':<16}   {status}")
        if hint:
            emit(f"  {'':<16}   {hint}")
    emit()

    emit(rule("SUBSYSTEMS"))
    for name in _SUBSYSTEMS:
        try:
            importlib.import_module(name)
        except Exception as error:
            emit(f"  {name:<22} NOT IMPORTABLE: {type(error).__name__}: {error}")
        else:
            emit(f"  {name:<22} ok")
    emit()

    if arguments.metrics:
        emit(rule("EVALUATION METRICS"))
        try:
            from avgen.eval import describe_metrics, learned_metric_summary

            for name, description in describe_metrics().items():
                emit(f"  {name:<24} {description}")
            emit()
            emit("  gated (need a pretrained backend):")
            emit(learned_metric_summary())
        except Exception as error:
            emit(f"  unavailable: {error}")
        emit()

    emit(rule("NEXT"))
    emit(
        "  avgen plan --world-size 8 --seq-len 16384 --params 3e8 "
        "--depth 24 --width 1536"
    )
    emit("  avgen train --config configs/train/smoke_cpu.yaml")
    return 0
