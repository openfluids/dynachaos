"""Test that Rust and Python implementations produce identical results.

These tests force both paths and compare outputs to ensure the Rust
acceleration is a transparent drop-in.
"""

import os

import numpy as np
import pytest
from conftest import logistic_series

try:
    from dynachaos._rust import diagonal_lines  # noqa: F401

    _HAS_RUST = not os.environ.get("DYNACHAOS_NO_RUST")
except ImportError:
    _HAS_RUST = False

needs_rust = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")


@needs_rust
class TestRecurrenceParity:
    """Verify Rust and Python _diagonal_lines / _vertical_lines agree."""

    def _recurrence_matrix(self):
        t = np.linspace(0.0, 40.0, 400)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        from dynachaos.diagnostics.recurrence import recurrence_matrix

        R, _ = recurrence_matrix(traj, percentile=8)
        return R

    def test_diagonal_lines_parity(self):
        R = self._recurrence_matrix()
        from dynachaos._rust import diagonal_lines as rust_diag

        # Python path (bypass Rust)
        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        rec_mod._RUST_AVAILABLE = False
        py_result = rec_mod._diagonal_lines(R, l_min=2)
        rec_mod._RUST_AVAILABLE = old_flag

        # Rust path
        rs_result = np.asarray(rust_diag(R, l_min=2))

        np.testing.assert_array_equal(sorted(py_result), sorted(rs_result))

    def test_vertical_lines_parity(self):
        R = self._recurrence_matrix()
        from dynachaos._rust import vertical_lines as rust_vert
        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        rec_mod._RUST_AVAILABLE = False
        py_result = rec_mod._vertical_lines(R, v_min=2)
        rec_mod._RUST_AVAILABLE = old_flag

        rs_result = np.asarray(rust_vert(R, v_min=2))

        np.testing.assert_array_equal(sorted(py_result), sorted(rs_result))

    def test_rqa_parity(self):
        """Full RQA pipeline should give same results via either path."""
        R = self._recurrence_matrix()
        from dynachaos.diagnostics import recurrence as rec_mod

        # Rust path (only works when Rust is loaded)
        rqa_rust = rec_mod.rqa(R)

        # Python path
        old_flag = rec_mod._RUST_AVAILABLE
        rec_mod._RUST_AVAILABLE = False
        rqa_python = rec_mod.rqa(R)
        rec_mod._RUST_AVAILABLE = old_flag

        for key in rqa_rust:
            assert rqa_rust[key] == pytest.approx(rqa_python[key], abs=1e-12), (
                f"RQA[{key}] mismatch: rust={rqa_rust[key]}, python={rqa_python[key]}"
            )


@needs_rust
class TestPermutationParity:
    """Verify Rust and Python ordinal_distribution agree."""

    def test_ordinal_distribution_parity(self):
        series = logistic_series(n=5000)
        from dynachaos._rust import ordinal_distribution as rust_ord

        # Python path
        from dynachaos.diagnostics import permutation as perm_mod

        old_flag = perm_mod._RUST_AVAILABLE
        perm_mod._RUST_AVAILABLE = False
        py_probs, py_total = perm_mod.ordinal_distribution(series, d=5, tau=1)
        perm_mod._RUST_AVAILABLE = old_flag

        # Rust path
        rs_counts, rs_total = rust_ord(series, d=5, tau=1)
        rs_counts = np.asarray(rs_counts)

        assert py_total == rs_total

        # Compare: reconstruct probabilities from Rust counts
        from dynachaos.diagnostics.permutation import _lehmer_to_permutation

        rs_probs = {}
        for idx in np.nonzero(rs_counts)[0]:
            perm = _lehmer_to_permutation(int(idx), 5)
            rs_probs[perm] = int(rs_counts[idx]) / rs_total

        # Same set of patterns
        assert set(py_probs.keys()) == set(rs_probs.keys())

        # Same probabilities
        for perm in py_probs:
            assert py_probs[perm] == pytest.approx(rs_probs[perm], abs=1e-12)

    @pytest.mark.parametrize(
        ("d", "tau", "message"),
        [
            (0, 1, "d must be >= 2"),
            (1, 1, "d must be >= 2"),
            (11, 1, "d must be <= 10"),
            (2, 0, "tau must be >= 1"),
            (5, 1, "time series is too short"),
        ],
    )
    def test_ordinal_distribution_rejects_invalid_direct_rust_inputs(self, d, tau, message):
        from dynachaos._rust import ordinal_distribution as rust_ord

        with pytest.raises(ValueError, match=message):
            rust_ord(np.arange(3.0), d=d, tau=tau)

    def test_permutation_entropy_parity(self):
        """Full permutation entropy should agree regardless of backend."""
        series = logistic_series(n=5000)
        from dynachaos.diagnostics import permutation as perm_mod

        # Rust path (Rust is loaded, flag is True)
        h_rust = perm_mod.permutation_entropy(series, d=5)

        # Python path
        old_flag = perm_mod._RUST_AVAILABLE
        perm_mod._RUST_AVAILABLE = False
        h_python = perm_mod.permutation_entropy(series, d=5)
        perm_mod._RUST_AVAILABLE = old_flag

        assert h_rust == pytest.approx(h_python, abs=1e-10)

    def test_complexity_entropy_parity(self):
        """Complexity-entropy plane should agree regardless of backend."""
        series = logistic_series(n=5000)
        from dynachaos.diagnostics import permutation as perm_mod

        # Rust path (Rust is loaded, flag is True)
        h_rust, c_rust = perm_mod.complexity_entropy(series, d=5)

        # Python path
        old_flag = perm_mod._RUST_AVAILABLE
        perm_mod._RUST_AVAILABLE = False
        h_python, c_python = perm_mod.complexity_entropy(series, d=5)
        perm_mod._RUST_AVAILABLE = old_flag

        assert h_rust == pytest.approx(h_python, abs=1e-10)
        assert c_rust == pytest.approx(c_python, abs=1e-10)


@needs_rust
class TestAMIParity:
    """Verify Rust and Python AMI agree."""

    def test_ami_parity(self):
        series = logistic_series(n=3000)
        from dynachaos.diagnostics import embedding as emb_mod

        # Rust path
        old_flag = emb_mod._RUST_AVAILABLE
        emb_mod._RUST_AVAILABLE = True
        _, I_rust = emb_mod.average_mutual_information(series, tau_max=30, n_bins=32)
        emb_mod._RUST_AVAILABLE = old_flag

        # Python path
        I_python = emb_mod._ami_python(series, tau_max=30, n_bins=32)

        np.testing.assert_allclose(
            I_rust, I_python, atol=1e-10, err_msg="AMI Rust vs Python mismatch"
        )


# TestCaoParity removed: Rust cao_statistic disabled (scipy cKDTree is 70x faster).


@needs_rust
class TestCaoSelectorParity:
    """Verify Rust and Python Cao selector agree."""

    @pytest.mark.parametrize(
        ("e1", "kwargs", "expected"),
        [
            (
                np.array([0.0024, 0.0559, 0.2308, 1.0013, 0.9835, 0.9979, 1.0000, 1.0000]),
                dict(
                    near_one_lower=0.97,
                    near_one_upper=1.03,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                4,
            ),
            (
                np.array([0.08, 0.20, 0.35, 0.70, 0.96, 0.93]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=4,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                5,
            ),
            (
                np.array([0.10, 0.20, 0.30, 0.40]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                4,
            ),
            (
                np.array([np.nan, np.inf, np.nan]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=3,
                    max_dim=15,
                ),
                3,
            ),
            # NaN mid-plateau: window must be rejected identically by both paths
            (
                np.array([0.50, 0.98, np.nan, 0.99, 1.00, 1.00, 1.00]),
                dict(
                    near_one_lower=0.95,
                    near_one_upper=1.05,
                    saturation_tol=0.02,
                    plateau_span=3,
                    smoothing_window=1,
                    min_dim=2,
                    max_dim=15,
                ),
                4,
            ),
        ],
    )
    def test_select_dimension_cao_parity(self, e1, kwargs, expected):
        from dynachaos._rust import select_dimension_cao as rust_selector
        from dynachaos.diagnostics import embedding as emb_mod

        e1 = np.asarray(e1, dtype=np.float64)

        d_rust = int(rust_selector(e1, **kwargs))

        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            d_python = int(emb_mod.select_dimension_cao(e1, **kwargs))
        finally:
            emb_mod._select_dimension_cao_rs = old_selector

        assert d_rust == d_python == expected


# TestFNNParity removed: Rust fnn_statistic disabled (scipy cKDTree is 100x faster).


@needs_rust
class TestCorrelationCountsParity:
    """Verify Rust and Python correlation integral agree."""

    def test_correlation_counts_parity(self):
        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 15)

        from dynachaos.diagnostics import correlation as corr_mod

        # Rust path
        old_flag = corr_mod._RUST_AVAILABLE
        corr_mod._RUST_AVAILABLE = True
        C_rust = corr_mod.correlation_integral(traj, r_values, theiler_window=5, norm="chebyshev")
        corr_mod._RUST_AVAILABLE = old_flag

        # Python path
        corr_mod._RUST_AVAILABLE = False
        C_py = corr_mod.correlation_integral(traj, r_values, theiler_window=5, norm="chebyshev")
        corr_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_allclose(
            C_rust, C_py, atol=1e-10, err_msg="Correlation integral Rust vs Python mismatch"
        )


class TestDiscreteMap:
    """Test the new DiscreteMap convenience class."""

    def test_logistic_trajectory(self):
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=1.99)
        traj = lm.trajectory(x0=0.1, n_iter=100, n_transient=50)
        assert traj.shape == (100,)
        assert np.all(np.isfinite(traj))

    def test_logistic_lyapunov(self):
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=1.99)
        lam = lm.lyapunov(x0=0.1, n_iter=50_000, n_transient=5_000)
        assert lam > 0.5  # chaotic

    def test_no_jacobian_raises(self):
        from dynachaos.maps.base import DiscreteMap

        m = DiscreteMap(f=lambda x: 1 - 1.99 * x * x, name="bare")
        with pytest.raises(ValueError, match="No derivative"):
            m.lyapunov(x0=0.1)

    def test_repr(self):
        from dynachaos.maps.base import LogisticMap

        lm = LogisticMap(a=1.5)
        assert "Logistic" in repr(lm)


class TestViz:
    """Smoke tests for viz subpackage (non-interactive)."""

    def test_bifurcation_import(self):
        from dynachaos.viz import bifurcation_diagram

        assert callable(bifurcation_diagram)

    def test_cobweb_import(self):
        from dynachaos.viz import cobweb_diagram

        assert callable(cobweb_diagram)

    def test_return_map_import(self):
        from dynachaos.viz import return_map_plot

        assert callable(return_map_plot)


class TestVersion:
    def test_version_string(self):
        import dynachaos

        assert dynachaos.__version__ == "0.2.0"
