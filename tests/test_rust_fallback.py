"""Test that Rust and Python implementations produce identical results.

These tests force both paths and compare outputs to ensure the Rust
acceleration is a transparent drop-in.
"""

import os
import sys

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

    @pytest.mark.parametrize(
        ("mask", "min_length"),
        [
            ([], 2),
            ([False, False, False], 2),
            ([True, False, True, True, False, True, True, True], 2),
            ([True, True, False, True, False, True, True], 1),
            ([False, True, True, True], 3),
        ],
    )
    def test_count_line_lengths_parity(self, mask, min_length):
        from dynachaos._rust import count_line_lengths as rust_line_lengths
        from dynachaos.diagnostics import recurrence as rec_mod

        mask_array = np.asarray(mask, dtype=np.bool_)

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            python_result = rec_mod._line_lengths(mask_array, min_length)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag
        rust_result = np.asarray(rust_line_lengths(mask_array, min_length))

        np.testing.assert_array_equal(rust_result, np.asarray(python_result, dtype=np.int64))

    def test_count_line_lengths_rejects_invalid_min_length(self):
        from dynachaos._rust import count_line_lengths as rust_line_lengths

        with pytest.raises(ValueError, match="min_length"):
            rust_line_lengths(np.array([True, False], dtype=np.bool_), 0)

    def test_streaming_rqa_uses_line_scanner_transparently(self):
        from dynachaos.diagnostics import recurrence as rec_mod

        t = np.linspace(0.0, 18.0, 120)
        traj = np.column_stack([np.sin(t), np.cos(1.4 * t)])

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = True
            rust_stats = rec_mod.rqa_from_trajectory(
                traj, percentile=7, metric="chebyshev", l_min=2, v_min=3
            )

            rec_mod._RUST_AVAILABLE = False
            python_stats = rec_mod.rqa_from_trajectory(
                traj, percentile=7, metric="chebyshev", l_min=2, v_min=3
            )
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        assert rust_stats.keys() == python_stats.keys()
        for key, value in python_stats.items():
            assert rust_stats[key] == pytest.approx(value)

    @pytest.mark.parametrize(
        ("function_name", "kwargs", "message"),
        [
            ("diagonal_lines", {"l_min": 0}, "l_min"),
            ("vertical_lines", {"v_min": 0}, "v_min"),
            ("diagonal_lines", {}, "square"),
            ("vertical_lines", {}, "square"),
        ],
    )
    def test_direct_rust_recurrence_rejects_invalid_inputs(self, function_name, kwargs, message):
        import dynachaos._rust as rust_mod

        rmat = np.ones((2, 3), dtype=bool)
        if "l_min" in kwargs or "v_min" in kwargs:
            rmat = np.eye(3, dtype=bool)

        with pytest.raises(ValueError, match=message):
            getattr(rust_mod, function_name)(rmat, **kwargs)

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

    def test_ami_direct_rust_rejects_invalid_inputs(self):
        from dynachaos._rust import ami_histogram

        with pytest.raises(ValueError, match="at least two"):
            ami_histogram(np.array([1.0]), tau_max=3, n_bins=4)
        with pytest.raises(ValueError, match="finite values"):
            ami_histogram(np.array([1.0, np.nan, 2.0]), tau_max=3, n_bins=4)

    def test_ami_direct_rust_constant_series_returns_zero(self):
        from dynachaos._rust import ami_histogram

        mi = ami_histogram(np.ones(10), tau_max=5, n_bins=4)

        np.testing.assert_array_equal(mi, np.zeros(5))


# TestCaoParity omitted: Cao statistics remain in Python/SciPy.


@needs_rust
def test_disabled_embedding_statistics_are_not_exported():
    import dynachaos._rust as rust_mod

    assert not hasattr(rust_mod, "cao_statistic")
    assert not hasattr(rust_mod, "fnn_statistic")
    assert hasattr(rust_mod, "select_dimension_cao")


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


# TestFNNParity omitted: FNN statistics remain in Python/SciPy.


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

    def test_correlation_counts_huge_theiler_window_has_no_pairs(self):
        from dynachaos._rust import correlation_counts

        traj = np.arange(12.0).reshape(6, 2)
        r_values = np.array([0.1, 10.0], dtype=np.float64)

        counts = np.asarray(correlation_counts(traj, r_values, sys.maxsize, True))

        np.testing.assert_array_equal(counts, np.zeros_like(r_values, dtype=np.int64))


@needs_rust
class TestEntropyRustBoundaries:
    def test_apen_counts_uses_inclusive_self_matches(self):
        from dynachaos._rust import apen_counts

        traj = np.array(
            [
                [0.0, 0.1],
                [0.1, 0.2],
                [0.2, 0.4],
                [0.4, 0.45],
            ],
            dtype=np.float64,
        )

        counts = np.asarray(apen_counts(traj, 0.2))

        np.testing.assert_array_equal(counts, np.array([2, 3, 3, 2], dtype=np.int64))

    def test_apen_counts_rejects_invalid_r(self):
        from dynachaos._rust import apen_counts

        with pytest.raises(ValueError, match="r must be positive"):
            apen_counts(np.arange(6.0).reshape(3, 2), 0.0)

    def test_fuzzy_entropy_sum_huge_theiler_window_has_no_pairs(self):
        from dynachaos._rust import fuzzy_entropy_sum

        traj = np.arange(12.0).reshape(6, 2)

        assert fuzzy_entropy_sum(traj, 1.0, 2, sys.maxsize) == 0.0


@needs_rust
class TestCMLJacobianParity:
    @pytest.mark.parametrize("L", [1, 3, 5])
    def test_cml_jacobian_logistic_direct_rust_matches_python_path(self, L):
        from dynachaos._rust import cml_jacobian_logistic as rust_jacobian
        from dynachaos.cml import primitives as cml_mod

        x = np.array([0.125, -0.25, 0.375, -0.5, 0.625], dtype=np.float64)
        a = 1.73
        eps = 0.31

        old_flag = cml_mod._RUST_AVAILABLE
        cml_mod._RUST_AVAILABLE = False
        try:
            py_jacobian = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        rust_jacobian_flat = np.asarray(rust_jacobian(x, a=a, eps=eps, L=L))
        rust_jacobian_matrix = rust_jacobian_flat.reshape((L, L))

        np.testing.assert_allclose(rust_jacobian_matrix, py_jacobian, atol=0.0, rtol=0.0)

    def test_cml_jacobian_public_dispatcher_matches_python_path(self):
        from dynachaos.cml import primitives as cml_mod

        x = np.array([0.2, -0.1, 0.4, -0.3], dtype=np.float64)
        a = 1.91
        eps = 0.27
        L = len(x)

        old_flag = cml_mod._RUST_AVAILABLE
        try:
            cml_mod._RUST_AVAILABLE = True
            rust_path = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)

            cml_mod._RUST_AVAILABLE = False
            python_path = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_allclose(rust_path, python_path, atol=0.0, rtol=0.0)

    def test_cml_jacobian_logistic_rejects_invalid_l(self):
        from dynachaos._rust import cml_jacobian_logistic as rust_jacobian

        x = np.array([0.1, -0.2, 0.3], dtype=np.float64)

        with pytest.raises(ValueError, match="L must satisfy"):
            rust_jacobian(x, 1.5, 0.2, 0)

        with pytest.raises(ValueError, match="L must satisfy"):
            rust_jacobian(x, 1.5, 0.2, len(x) + 1)

    def test_cml_lyapunov_density_inner_loop_parity(self):
        rust_density = self._small_lyapunov_density(use_rust=True)
        python_density = self._small_lyapunov_density(use_rust=False)

        np.testing.assert_allclose(rust_density, python_density, atol=1e-14, rtol=1e-14)

    def _small_lyapunov_density(self, use_rust):
        from dynachaos.cml import primitives as cml_mod

        N = 6
        eps = 0.3
        a_values = np.array([1.55, 1.82])
        L_values = np.array([2, 4, 6])
        n_transient = 5
        n_iter = 10
        rng = np.random.default_rng(123)

        density = np.empty((len(a_values), len(L_values)))
        old_flag = cml_mod._RUST_AVAILABLE
        cml_mod._RUST_AVAILABLE = use_rust
        try:
            for ia, a in enumerate(a_values):
                x = rng.uniform(-0.5, 0.5, N)
                for _ in range(n_transient):
                    x = cml_mod.cml_step_logistic(x, a, eps)

                for iL, L in enumerate(L_values):
                    x_run = x.copy()
                    v = rng.standard_normal(L)
                    v /= np.linalg.norm(v)

                    log_sum = 0.0
                    for _ in range(n_iter):
                        jacobian = cml_mod.cml_jacobian_subblock_logistic(x_run, a, eps, L)
                        v = jacobian @ v
                        norm_v = np.linalg.norm(v)
                        if norm_v > 0:
                            log_sum += np.log(norm_v)
                            v /= norm_v
                        else:
                            log_sum += -100.0
                            v = rng.standard_normal(L)
                            v /= np.linalg.norm(v)
                        x_run = cml_mod.cml_step_logistic(x_run, a, eps)

                    density[ia, iL] = log_sum / n_iter / L
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        return density


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
