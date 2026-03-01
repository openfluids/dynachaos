"""Path resolution and I/O helpers for generated figure/data artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


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


def safe_load(path):
    """Load .npz without deserializing arbitrary objects."""
    return np.load(path, allow_pickle = False)
