"""Shared test fixtures and helpers."""

import platform
import sys

import numpy as np
import pytest

# The committed figure caches under figures/ were computed on Linux x86_64.
#
# They cannot be reproduced elementwise on another architecture. Most of the
# sections integrate chaotic maps and coupled map lattices, where a last-bit
# difference in the first arithmetic operation is amplified exponentially: on
# macOS arm64 the recomputed sec12_intermittency cache differs from the
# committed one by a relative 4e4, and sec08_sti differs in 100% of elements.
# That is the systems behaving correctly, not a defect, and no tolerance can
# accommodate it.
#
# So the elementwise comparison is asserted only on the architecture that
# produced the caches. Elsewhere the pipelines are still executed and their
# output is still checked for shape, dtype and finiteness, which is what
# catches real breakage (crashes, NaN, contract drift) on other platforms.
REFERENCE_PLATFORM = "Linux x86_64"


def is_reference_platform() -> bool:
    """True on the architecture the committed figure caches were generated on."""
    return sys.platform.startswith("linux") and platform.machine() == "x86_64"


requires_reference_platform = pytest.mark.skipif(
    not is_reference_platform(),
    reason=(
        f"committed figure caches are {REFERENCE_PLATFORM} artifacts; chaotic "
        "sections cannot reproduce them elementwise on another architecture"
    ),
)


def assert_threshold_well_separated(distances, eps, min_margin=1e-9):
    """Assert no distance sits close enough to ``eps`` to flip between platforms.

    Tests that pin goldens derived from a thresholded distance matrix are only
    portable while no distance lies near the threshold. Pairwise distances differ
    in their last bits across BLAS implementations and architectures, so a
    distance within ~1e-15 of ``eps`` can fall on either side depending on the
    machine, changing the extracted lines and the pinned values.

    This makes that precondition explicit and enforced rather than accidental:
    if someone retunes the trajectory or the percentile into a degenerate
    configuration, the failure says so instead of appearing later as an
    unexplained mismatch on one platform's CI runner.

    The circle trajectory in TestRecurrenceParity is the cautionary case — its
    distances land exactly on the threshold (margin 0.0), which is why it needed
    quantisation instead of separation.
    """
    distances = np.asarray(distances)
    margin = float(np.min(np.abs(distances - eps)))
    assert margin > min_margin, (
        f"threshold eps={eps!r} is only {margin:.3e} from the nearest pairwise "
        f"distance (need > {min_margin:.0e}). Goldens derived from this "
        f"threshold are not portable across architectures."
    )
    return margin


def logistic_series(n=2000, a=1.99, burn=500):
    """Generate a logistic map time series for testing."""
    x = 0.123456789
    series = np.empty(n)
    for i in range(n + burn):
        x = 1.0 - a * x * x
        if i >= burn:
            series[i - burn] = x
    return series


@pytest.fixture
def chaotic_series():
    """A chaotic logistic map series (a=1.99, 5000 points)."""
    return logistic_series(n=5000, a=1.99, burn=2000)
