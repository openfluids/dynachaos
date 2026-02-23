"""Path resolution for generated figure/data artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def output_root() -> Path:
    """Return the base output directory for generated artifacts.

    Precedence:
    1. ``DYNACHAOS_OUTPUT_ROOT`` environment variable
    2. ``<cwd>/figures``
    """
    env_root = os.environ.get("DYNACHAOS_OUTPUT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.cwd() / "figures").resolve()


def section_dir(section_name: str) -> Path:
    """Return the output directory for a paper section."""
    return output_root() / section_name
