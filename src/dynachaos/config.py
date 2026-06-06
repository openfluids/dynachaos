"""Central configuration for dynachaos."""

from __future__ import annotations

import os

# Single source of truth for figure theme across all scripts.
DEFAULT_FIGURE_THEME = "signature"

# Optional runtime override for quick comparisons without editing code.
FIGURE_THEME_ENV_VAR = "DYNACHAOS_THEME"


def get_figure_theme() -> str:
    """Return active figure theme ID."""
    return os.environ.get(FIGURE_THEME_ENV_VAR, DEFAULT_FIGURE_THEME)
