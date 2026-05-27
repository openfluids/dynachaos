"""Tests for embedding parameter selection (AMI, Cao, FNN).

Correctness tests use known-dimension systems:
- Sinusoid: AMI first minimum near quarter-period
- Circle trajectory: FNN should vanish at d=2
- Delayed logistic: Cao E1 should saturate near d=2-3
"""

import importlib

import numpy as np
import pytest
from conftest import logistic_series

from dynachaos.diagnostics.embedding import (
    average_mutual_information,
    cao_method,
    false_nearest_neighbors,
    optimal_delay,
    optimal_dimension,
    select_dimension_cao,
)


class TestAMI:
    def test_ami_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 20 * np.pi, 2000)) + 0.01 * rng.standard_normal(2000)
        taus, mi = average_mutual_information(x, tau_max=50, n_bins=64)
        assert taus.shape == (50,)
        assert mi.shape == (50,)
        assert np.all(mi >= 0)

    def test_ami_sinusoid_first_minimum(self):
        """Sinusoid with period 20: AMI first minimum near tau=5 (quarter-period)."""
        t = np.arange(2000, dtype=np.float64)
        x = np.sin(2 * np.pi * 0.05 * t)
        tau_opt = optimal_delay(x, tau_max=50)
        # Quarter period = 5, allow some slack
        assert 3 <= tau_opt <= 7, f"Expected tau_opt near 5, got {tau_opt}"

    def test_ami_decreasing_initially(self):
        """AMI should generally decrease from tau=1."""
        x = logistic_series(n=5000)
        _, mi = average_mutual_information(x, tau_max=30)
        # First few values should decrease (MI drops with delay for chaotic series)
        assert mi[0] > mi[2], "AMI should decrease initially for chaotic series"


class TestCao:
    def test_cao_returns_correct_shape(self):
        x = logistic_series(n=3000)
        E1, E2 = cao_method(x, tau=1, d_max=10)
        assert E1.shape == (9,)  # d_max - 1
        assert E2.shape == (9,)

    def test_cao_logistic_saturation(self):
        """Logistic map (1D map -> d_opt=1 or 2): E1 should saturate quickly."""
        x = logistic_series(n=5000, a=1.99)
        E1, E2 = cao_method(x, tau=1, d_max=10)
        # E1 should be near 1 for d >= 2
        assert E1[2] > 0.9, f"E1(3) should be near 1, got {E1[2]}"

    def test_cao_deterministic_e2(self):
        """E2 should deviate from 1 for deterministic series."""
        x = logistic_series(n=5000, a=1.99)
        _, E2 = cao_method(x, tau=1, d_max=8)
        # For a deterministic system, E2 should not all be 1
        assert not np.allclose(E2, 1.0, atol=0.1), (
            "E2 should deviate from 1 for deterministic signal"
        )


class TestFNN:
    def test_fnn_returns_correct_shape(self):
        x = logistic_series(n=3000)
        f1, f2, f3 = false_nearest_neighbors(x, tau=1, d_max=8)
        assert f1.shape == (8,)
        assert f2.shape == (8,)
        assert f3.shape == (8,)

    def test_fnn_circle_low_at_d2(self):
        """Circle in 2D (d=1 intrinsic): FNN should vanish by d=2."""
        t = np.linspace(0, 100, 3000)
        x = np.sin(t)
        f1, f2, f3 = false_nearest_neighbors(x, tau=15, d_max=6)
        # FNN fraction should be low at d=2
        assert f3[1] < 0.15, f"FNN(d=2) should be low, got {f3[1]}"

    def test_fnn_bounded(self):
        """All FNN fractions should be in [0, 1]."""
        x = logistic_series(n=3000)
        f1, f2, f3 = false_nearest_neighbors(x, tau=1, d_max=6)
        for arr in (f1, f2, f3):
            valid = arr[np.isfinite(arr)]
            assert np.all(valid >= 0) and np.all(valid <= 1)


class TestOptimalDimension:
    def test_cao_selector_env_forced_python_backend(self, monkeypatch):
        import dynachaos.diagnostics.embedding as emb_mod

        monkeypatch.setenv("DYNACHAOS_NO_RUST", "1")
        emb_mod = importlib.reload(emb_mod)
        try:
            e1 = np.array(
                [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
                dtype=np.float64,
            )
            d = emb_mod.select_dimension_cao(
                e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2
            )
            assert emb_mod._RUST_AVAILABLE is False
            assert d == 4
        finally:
            monkeypatch.delenv("DYNACHAOS_NO_RUST", raising=False)
            importlib.reload(emb_mod)

    def test_cao_selector_python_fallback_path(self):
        from dynachaos.diagnostics import embedding as emb_mod

        e1 = np.array(
            [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
            dtype=np.float64,
        )
        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            d = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
        finally:
            emb_mod._select_dimension_cao_rs = old_selector

        assert d == 4

    def test_cao_selector_onset_not_tail(self):
        e1 = np.array(
            [0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000],
            dtype=np.float64,
        )
        d = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
        assert d == 4

    def test_cao_selector_ignores_low_dim_artifacts(self):
        e1 = np.array(
            [0.0010, 0.1340, 0.1587, 0.8184, 0.8658, 0.9646, 0.9961, 0.9948, 0.9989],
            dtype=np.float64,
        )
        d = select_dimension_cao(e1, near_one_lower=0.97, near_one_upper=1.03, min_dim=2)
        assert d >= 6

    def test_cao_method(self):
        x = logistic_series(n=5000, a=1.99)
        d = optimal_dimension(x, tau=1, d_max=10, method="cao")
        assert 1 <= d <= 4, f"Expected d_opt near 1-3 for logistic, got {d}"

    def test_cao_legacy_method(self):
        x = logistic_series(n=5000, a=1.99)
        d = optimal_dimension(x, tau=1, d_max=10, method="cao_legacy")
        assert 1 <= d <= 4, f"Expected d_opt near 1-3 for logistic, got {d}"

    def test_fnn_method(self):
        x = logistic_series(n=5000, a=1.99)
        d = optimal_dimension(x, tau=1, d_max=10, method="fnn")
        assert 1 <= d <= 5, f"Expected d_opt near 1-3 for logistic, got {d}"

    def test_cao_rejects_d_max_below_2(self):
        series = logistic_series(n=500)
        with pytest.raises(ValueError, match="d_max must be >= 2"):
            optimal_dimension(series, tau=1, d_max=1, method="cao")

    def test_cao_legacy_rejects_d_max_below_2(self):
        series = logistic_series(n=500)
        with pytest.raises(ValueError, match="d_max must be >= 2"):
            optimal_dimension(series, tau=1, d_max=1, method="cao_legacy")

    def test_cao_d_max_2_succeeds(self):
        """Boundary: d_max=2 is the minimum valid value for Cao."""
        series = logistic_series(n=500)
        d = optimal_dimension(series, tau=1, d_max=2, method="cao")
        assert 1 <= d <= 2

    def test_d_max_zero_rejects_all_methods(self):
        series = logistic_series(n=500)
        with pytest.raises(ValueError, match="d_max must be >= 1"):
            optimal_dimension(series, tau=1, d_max=0, method="fnn")

    def test_invalid_method_raises(self):
        x = logistic_series(n=1000)
        with pytest.raises(ValueError, match="Unknown method"):
            optimal_dimension(x, tau=1, method="invalid")


class TestCorrelationIntegralImproved:
    def test_backward_compat(self):
        """Old-style call (no theiler_window, no norm) should still work."""
        from dynachaos.diagnostics.correlation import correlation_integral

        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 10)
        C = correlation_integral(traj, r_values)
        assert C.shape == (10,)
        assert np.all(C >= 0)
        assert np.all(C <= 1)

    def test_theiler_window_reduces_counts(self):
        """With Theiler window, counts should be <= without."""
        from dynachaos.diagnostics.correlation import correlation_integral

        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 10)
        C0 = correlation_integral(traj, r_values, theiler_window=0)
        C10 = correlation_integral(traj, r_values, theiler_window=10)
        # With theiler window, C should generally change (fewer pairs)
        assert not np.allclose(C0, C10), "Theiler window should affect C(r)"

    def test_correlation_integral_is_monotone_in_radius(self):
        from dynachaos.diagnostics.correlation import correlation_integral

        rng = np.random.default_rng(123)
        traj = rng.normal(size=(200, 2))
        r_values = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=np.float64)

        C = correlation_integral(traj, r_values, theiler_window=3)

        assert np.all(np.diff(C) >= 0.0)

    def test_valid_pair_count_decreases_with_theiler_window(self):
        from dynachaos.diagnostics.correlation import _valid_pair_count

        counts = np.array([_valid_pair_count(20, w) for w in range(6)])

        assert np.all(np.diff(counts) < 0)

    def test_invalid_norm_raises(self):
        from dynachaos.diagnostics.correlation import correlation_integral

        with pytest.raises(ValueError, match="norm must be one of"):
            correlation_integral(np.arange(5), [1.0], norm="manhattan")

    def test_negative_theiler_window_raises(self):
        from dynachaos.diagnostics.correlation import correlation_integral

        with pytest.raises(ValueError, match="theiler_window must be >= 0"):
            correlation_integral(np.arange(5), [1.0], theiler_window=-1)

    def test_constant_trajectory_returns_nan_dimension(self):
        from dynachaos.diagnostics.correlation import correlation_dimension

        D2, r_values, C_values, slopes, scaling = correlation_dimension(np.ones((10, 2)))

        assert np.isnan(D2)
        assert r_values.shape == (0,)
        assert C_values.shape == (0,)
        assert slopes.shape == (0,)
        assert scaling.shape == (0,)

    def test_too_short_trajectory_returns_nan_dimension(self):
        from dynachaos.diagnostics.correlation import correlation_dimension

        for traj in (np.array([]), np.array([1.0])):
            D2, r_values, C_values, slopes, scaling = correlation_dimension(traj)

            assert np.isnan(D2)
            assert r_values.shape == (0,)
            assert C_values.shape == (0,)
            assert slopes.shape == (0,)
            assert scaling.shape == (0,)

    def test_correlation_dimension_circle(self):
        """Circle (D=1) should give D2 ~ 1 with improved G-P."""
        from dynachaos.diagnostics.correlation import correlation_dimension

        t = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        D2, _, _, _, _ = correlation_dimension(traj)
        assert 0.7 < D2 < 1.4, f"Expected D2 ~ 1, got {D2}"
