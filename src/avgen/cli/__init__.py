"""The ``avgen`` command line.

``argparse`` and nothing else. click and typer are both nicer to write against;
neither is worth a dependency in a path that has to work on a cluster login node
with torch and no optional extras installed.

The entry point is :func:`avgen.cli.main.main`, registered as the ``avgen``
console script. Every subcommand lives in its own module and imports its
subsystem lazily inside its handler, so ``avgen --help``, ``avgen info``, and
``avgen plan`` keep working when a subsystem does not import — which is exactly
when you need those three commands.
"""

from avgen.cli.main import build_parser, main

__all__ = ["build_parser", "main"]
