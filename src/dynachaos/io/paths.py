"""Path resolution and I/O helpers for generated figure/data artifacts."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

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
    return np.load(path, allow_pickle=False)


def load_or_compute_npz(
    npz_path: str | Path,
    section_name: str,
    compute_fn: Callable[[], Any],
    *,
    required_keys: Iterable[str] = (),
):
    """Load a cache file, recomputing it when missing or schema-stale."""
    path = Path(npz_path)
    keys = tuple(required_keys)
    try:
        data = safe_load(path)
        missing = tuple(key for key in keys if key not in data.files)
        if missing:
            data.close()
            missing_text = ", ".join(missing)
            print(f"Cache {path} missing keys ({missing_text}); recomputing {section_name}...")
            compute_fn()
            data = safe_load(path)
        else:
            print(f"Loaded {path}")
    except FileNotFoundError:
        print(f"Computing {section_name}...")
        compute_fn()
        data = safe_load(path)

    missing = tuple(key for key in keys if key not in data.files)
    if missing:
        data.close()
        missing_text = ", ".join(missing)
        raise KeyError(f"Cache {path} is missing required keys after compute: {missing_text}")
    return data
