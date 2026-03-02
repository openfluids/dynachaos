"""System-level utilities (memory, resource usage)."""

import resource
import sys


def get_rss_mb() -> float:
    """Return current process RSS in megabytes."""
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: ru_maxrss is in bytes; Linux: in KB
    return rss_raw / (1024 * 1024) if sys.platform == "darwin" else rss_raw / 1024
