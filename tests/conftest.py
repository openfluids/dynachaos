"""Shared test fixtures and helpers."""

import numpy as np
import pytest


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
