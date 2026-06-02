"""Test that Rust and Python implementations produce identical results.

These tests force both paths and compare outputs to ensure the Rust
acceleration is a transparent drop-in.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import logistic_series

try:
    import dynachaos._rust  # noqa: F401

    _RUST_IMPORTABLE = True
except ImportError:
    _RUST_IMPORTABLE = False

rust_extension = pytest.mark.skipif(not _RUST_IMPORTABLE, reason="compiled extension not built")
_NO_RUST_ENV = bool(os.environ.get("DYNACHAOS_NO_RUST"))
_GOLDEN_PATH = Path(__file__).with_name("data") / "rust_parity_goldens.npz"


def _golden(name):
    with np.load(_GOLDEN_PATH) as goldens:
        return goldens[name]


def _assert_golden(name, value, **kwargs):
    np.testing.assert_allclose(np.asarray(value), _golden(name), **kwargs)


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

        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            py_result = rec_mod._diagonal_lines(R, l_min=2)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray(sorted(py_result), dtype=np.int64),
            _golden("diag_lines"),
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import diagonal_lines as rust_diag

            rs_result = np.asarray(sorted(rust_diag(R, l_min=2)), dtype=np.int64)
            np.testing.assert_array_equal(rs_result, _golden("diag_lines"))

    def test_vertical_lines_parity(self):
        R = self._recurrence_matrix()
        from dynachaos.diagnostics import recurrence as rec_mod

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            py_result = rec_mod._vertical_lines(R, v_min=2)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray(sorted(py_result), dtype=np.int64),
            _golden("vert_lines"),
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import vertical_lines as rust_vert

            rs_result = np.asarray(sorted(rust_vert(R, v_min=2)), dtype=np.int64)
            np.testing.assert_array_equal(rs_result, _golden("vert_lines"))

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
    def test_count_line_lengths_parity(self, mask, min_length, request):
        from dynachaos.diagnostics import recurrence as rec_mod

        mask_array = np.asarray(mask, dtype=np.bool_)

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            python_result = rec_mod._line_lengths(mask_array, min_length)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag
        golden_name = f"line_lengths_{request.node.callspec.indices['mask']}"
        np.testing.assert_array_equal(
            np.asarray(python_result, dtype=np.int64),
            _golden(golden_name),
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import count_line_lengths as rust_line_lengths

            rust_result = np.asarray(rust_line_lengths(mask_array, min_length))
            np.testing.assert_array_equal(rust_result, _golden(golden_name))

    @rust_extension
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
            rec_mod._RUST_AVAILABLE = False
            python_stats = rec_mod.rqa_from_trajectory(
                traj, percentile=7, metric="chebyshev", l_min=2, v_min=3
            )
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray(sorted(python_stats)),
            _golden("streaming_rqa_keys"),
        )
        _assert_golden(
            "streaming_rqa_values",
            [python_stats[k] for k in sorted(python_stats)],
            atol=0.0,
            rtol=0.0,
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                rec_mod._RUST_AVAILABLE = True
                rust_stats = rec_mod.rqa_from_trajectory(
                    traj, percentile=7, metric="chebyshev", l_min=2, v_min=3
                )
            finally:
                rec_mod._RUST_AVAILABLE = old_flag

            np.testing.assert_array_equal(
                np.asarray(sorted(rust_stats)),
                _golden("streaming_rqa_keys"),
            )
            _assert_golden(
                "streaming_rqa_values",
                [rust_stats[k] for k in sorted(rust_stats)],
                atol=0.0,
                rtol=0.0,
            )

    @rust_extension
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

        old_flag = rec_mod._RUST_AVAILABLE
        try:
            rec_mod._RUST_AVAILABLE = False
            rqa_python = rec_mod.rqa(R)
        finally:
            rec_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(np.asarray(sorted(rqa_python)), _golden("rqa_keys"))
        _assert_golden(
            "rqa_values",
            [rqa_python[k] for k in sorted(rqa_python)],
            atol=1e-12,
            rtol=0.0,
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            rqa_rust = rec_mod.rqa(R)
            _assert_golden(
                "rqa_values",
                [rqa_rust[k] for k in sorted(rqa_rust)],
                atol=1e-12,
                rtol=0.0,
            )


class TestPermutationParity:
    """Verify Rust and Python ordinal_distribution agree."""

    def test_ordinal_distribution_parity(self):
        series = logistic_series(n=5000)

        from dynachaos.diagnostics import permutation as perm_mod

        old_flag = perm_mod._RUST_AVAILABLE
        try:
            perm_mod._RUST_AVAILABLE = False
            py_probs, py_total = perm_mod.ordinal_distribution(series, d=5, tau=1)
        finally:
            perm_mod._RUST_AVAILABLE = old_flag

        np.testing.assert_array_equal(
            np.asarray([py_total], dtype=np.int64),
            _golden("ordinal_total"),
        )
        np.testing.assert_array_equal(
            np.asarray(sorted(py_probs), dtype=np.int64),
            _golden("ordinal_indices"),
        )
        _assert_golden(
            "ordinal_probs",
            [py_probs[k] for k in sorted(py_probs)],
            atol=1e-12,
            rtol=0.0,
        )

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import ordinal_distribution as rust_ord
            from dynachaos.diagnostics.permutation import _lehmer_to_permutation

            rs_counts, rs_total = rust_ord(series, d=5, tau=1)
            rs_counts = np.asarray(rs_counts)
            np.testing.assert_array_equal(
                np.asarray([rs_total], dtype=np.int64),
                _golden("ordinal_total"),
            )

            rs_probs = {}
            for idx in np.nonzero(rs_counts)[0]:
                perm = _lehmer_to_permutation(int(idx), 5)
                rs_probs[perm] = int(rs_counts[idx]) / rs_total

            np.testing.assert_array_equal(
                np.asarray(sorted(rs_probs), dtype=np.int64),
                _golden("ordinal_indices"),
            )
            _assert_golden(
                "ordinal_probs",
                [rs_probs[k] for k in sorted(rs_probs)],
                atol=1e-12,
                rtol=0.0,
            )

    @rust_extension
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

        old_flag = perm_mod._RUST_AVAILABLE
        try:
            perm_mod._RUST_AVAILABLE = False
            h_python = perm_mod.permutation_entropy(series, d=5)
        finally:
            perm_mod._RUST_AVAILABLE = old_flag

        _assert_golden("permutation_entropy", [h_python], atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            h_rust = perm_mod.permutation_entropy(series, d=5)
            _assert_golden("permutation_entropy", [h_rust], atol=1e-10, rtol=0.0)

    def test_complexity_entropy_parity(self):
        """Complexity-entropy plane should agree regardless of backend."""
        series = logistic_series(n=5000)
        from dynachaos.diagnostics import permutation as perm_mod

        old_flag = perm_mod._RUST_AVAILABLE
        try:
            perm_mod._RUST_AVAILABLE = False
            h_python, c_python = perm_mod.complexity_entropy(series, d=5)
        finally:
            perm_mod._RUST_AVAILABLE = old_flag

        _assert_golden("complexity_entropy", [h_python, c_python], atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            h_rust, c_rust = perm_mod.complexity_entropy(series, d=5)
            _assert_golden("complexity_entropy", [h_rust, c_rust], atol=1e-10, rtol=0.0)


class TestAMIParity:
    """Verify Rust and Python AMI agree."""

    def test_ami_parity(self):
        series = logistic_series(n=3000)
        from dynachaos.diagnostics import embedding as emb_mod

        I_python = emb_mod._ami_python(series, tau_max=30, n_bins=32)

        _assert_golden("ami", I_python, atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            old_flag = emb_mod._RUST_AVAILABLE
            try:
                emb_mod._RUST_AVAILABLE = True
                _, I_rust = emb_mod.average_mutual_information(series, tau_max=30, n_bins=32)
            finally:
                emb_mod._RUST_AVAILABLE = old_flag
            _assert_golden("ami", I_rust, atol=1e-10, rtol=0.0)

    @rust_extension
    def test_ami_direct_rust_rejects_invalid_inputs(self):
        from dynachaos._rust import ami_histogram

        with pytest.raises(ValueError, match="at least two"):
            ami_histogram(np.array([1.0]), tau_max=3, n_bins=4)
        with pytest.raises(ValueError, match="finite values"):
            ami_histogram(np.array([1.0, np.nan, 2.0]), tau_max=3, n_bins=4)

    @rust_extension
    def test_ami_direct_rust_constant_series_returns_zero(self):
        from dynachaos._rust import ami_histogram

        mi = ami_histogram(np.ones(10), tau_max=5, n_bins=4)

        np.testing.assert_array_equal(mi, np.zeros(5))


# TestCaoParity omitted: Cao statistics remain in Python/SciPy.


@rust_extension
def test_disabled_embedding_statistics_are_not_exported():
    import dynachaos._rust as rust_mod

    assert not hasattr(rust_mod, "cao_statistic")
    assert not hasattr(rust_mod, "fnn_statistic")
    assert hasattr(rust_mod, "select_dimension_cao")


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
        from dynachaos.diagnostics import embedding as emb_mod

        e1 = np.asarray(e1, dtype=np.float64)

        old_selector = emb_mod._select_dimension_cao_rs
        emb_mod._select_dimension_cao_rs = None
        try:
            d_python = int(emb_mod.select_dimension_cao(e1, **kwargs))
        finally:
            emb_mod._select_dimension_cao_rs = old_selector

        assert d_python == expected

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import select_dimension_cao as rust_selector

            assert int(rust_selector(e1, **kwargs)) == expected


# TestFNNParity omitted: FNN statistics remain in Python/SciPy.


class TestCorrelationCountsParity:
    """Verify Rust and Python correlation integral agree."""

    def test_correlation_counts_parity(self):
        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        traj = np.column_stack([np.cos(t), np.sin(t)])
        r_values = np.logspace(-2, 0, 15)

        from dynachaos.diagnostics import correlation as corr_mod

        old_flag = corr_mod._RUST_AVAILABLE
        try:
            corr_mod._RUST_AVAILABLE = False
            C_py = corr_mod.correlation_integral(traj, r_values, theiler_window=5, norm="chebyshev")
        finally:
            corr_mod._RUST_AVAILABLE = old_flag

        _assert_golden("correlation_counts", C_py, atol=1e-10, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                corr_mod._RUST_AVAILABLE = True
                C_rust = corr_mod.correlation_integral(
                    traj, r_values, theiler_window=5, norm="chebyshev"
                )
            finally:
                corr_mod._RUST_AVAILABLE = old_flag
            _assert_golden("correlation_counts", C_rust, atol=1e-10, rtol=0.0)

    @rust_extension
    def test_correlation_counts_huge_theiler_window_has_no_pairs(self):
        from dynachaos._rust import correlation_counts

        traj = np.arange(12.0).reshape(6, 2)
        r_values = np.array([0.1, 10.0], dtype=np.float64)

        counts = np.asarray(correlation_counts(traj, r_values, sys.maxsize, True))

        np.testing.assert_array_equal(counts, np.zeros_like(r_values, dtype=np.int64))


@rust_extension
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


class TestCMLJacobianParity:
    @pytest.mark.parametrize("L", [1, 3, 5])
    def test_cml_jacobian_logistic_direct_rust_matches_python_path(self, L):
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

        _assert_golden(f"cml_jacobian_direct_{L}", py_jacobian, atol=0.0, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import cml_jacobian_logistic as rust_jacobian

            rust_jacobian_flat = np.asarray(rust_jacobian(x, a=a, eps=eps, L=L))
            rust_jacobian_matrix = rust_jacobian_flat.reshape((L, L))
            _assert_golden(f"cml_jacobian_direct_{L}", rust_jacobian_matrix, atol=0.0, rtol=0.0)

    def test_cml_jacobian_public_dispatcher_matches_python_path(self):
        from dynachaos.cml import primitives as cml_mod

        x = np.array([0.2, -0.1, 0.4, -0.3], dtype=np.float64)
        a = 1.91
        eps = 0.27
        L = len(x)

        old_flag = cml_mod._RUST_AVAILABLE
        try:
            cml_mod._RUST_AVAILABLE = False
            python_path = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
        finally:
            cml_mod._RUST_AVAILABLE = old_flag

        _assert_golden("cml_jacobian_public", python_path, atol=0.0, rtol=0.0)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                cml_mod._RUST_AVAILABLE = True
                rust_path = cml_mod.cml_jacobian_subblock_logistic(x, a, eps, L)
            finally:
                cml_mod._RUST_AVAILABLE = old_flag
            _assert_golden("cml_jacobian_public", rust_path, atol=0.0, rtol=0.0)

    @rust_extension
    def test_cml_jacobian_logistic_rejects_invalid_l(self):
        from dynachaos._rust import cml_jacobian_logistic as rust_jacobian

        x = np.array([0.1, -0.2, 0.3], dtype=np.float64)

        with pytest.raises(ValueError, match="L must satisfy"):
            rust_jacobian(x, 1.5, 0.2, 0)

        with pytest.raises(ValueError, match="L must satisfy"):
            rust_jacobian(x, 1.5, 0.2, len(x) + 1)

    def test_cml_lyapunov_density_inner_loop_parity(self):
        python_density = self._small_lyapunov_density(use_rust=False)

        _assert_golden("cml_lyapunov_density", python_density, atol=1e-14, rtol=1e-14)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            rust_density = self._small_lyapunov_density(use_rust=True)
            _assert_golden("cml_lyapunov_density", rust_density, atol=1e-14, rtol=1e-14)

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


class TestComovingLyapunovParity:
    """Verify specialized Rust and Python co-moving Lyapunov paths agree."""

    def test_comoving_lyapunov_logistic_kernel_parity(self):
        from dynachaos.diagnostics.comoving_lyapunov import _comoving_lyapunov_spectrum_python
        from dynachaos.maps.primitives import logistic, logistic_derivative

        a = 1.7
        eps = 0.3
        x_init = np.linspace(-0.75, 0.85, 12)
        v_values = np.array([-0.5, 0.0, 0.75], dtype=np.float64)
        n_iter = 37
        n_transient = 11

        python_result = _comoving_lyapunov_spectrum_python(
            lambda x: logistic(x, a),
            lambda x: logistic_derivative(x, a),
            lambda x: logistic(x, a),
            lambda x: logistic_derivative(x, a),
            eps,
            x_init,
            v_values,
            n_iter,
            n_transient,
        )
        _assert_golden("comoving_kernel", python_result, rtol=1e-12, atol=1e-12)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import comoving_lyapunov_logistic

            rust_result = np.asarray(
                comoving_lyapunov_logistic(
                    x_init,
                    v_values,
                    a,
                    eps,
                    n_iter,
                    n_transient,
                )
            )
            _assert_golden("comoving_kernel", rust_result, rtol=1e-12, atol=1e-12)

    def test_comoving_lyapunov_logistic_public_dispatch_parity(self):
        import importlib

        comoving_mod = importlib.import_module("dynachaos.diagnostics.comoving_lyapunov")

        kwargs = {
            "a": 1.85,
            "eps": 0.3,
            "N": 10,
            "v_values": np.array([-0.25, 0.25], dtype=np.float64),
            "n_iter": 25,
            "n_transient": 7,
            "x_init": np.linspace(-0.5, 0.6, 10),
        }
        old_flag = comoving_mod._RUST_AVAILABLE
        try:
            comoving_mod._RUST_AVAILABLE = False
            python_result = comoving_mod.comoving_lyapunov_spectrum_logistic(**kwargs)
        finally:
            comoving_mod._RUST_AVAILABLE = old_flag

        _assert_golden("comoving_public", python_result, rtol=1e-12, atol=1e-12)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                comoving_mod._RUST_AVAILABLE = True
                rust_result = comoving_mod.comoving_lyapunov_spectrum_logistic(**kwargs)
            finally:
                comoving_mod._RUST_AVAILABLE = old_flag
            _assert_golden("comoving_public", rust_result, rtol=1e-12, atol=1e-12)


class TestCoupledLogisticBasinsParity:
    """Verify Rust and Python coupled-logistic basin grids agree."""

    def test_coupled_logistic_basin_grid_parity(self):
        from dynachaos.maps.coupled_logistic import _basin_grid_python, _find_reference_orbit

        A = 1.35344
        D = 0.1
        n_grid = 5
        n_transient = 6
        reference_transient = 8
        period = 4

        ref_a = _find_reference_orbit(
            A,
            D,
            0.1,
            0.6,
            n_transient=reference_transient,
            period=period,
        )
        x_range = np.linspace(-1.0, 1.0, n_grid)
        y_range = np.linspace(-1.0, 1.0, n_grid)

        python_basin = _basin_grid_python(A, D, x_range, y_range, n_transient, ref_a)

        np.testing.assert_array_equal(python_basin, _golden("basin_grid"))

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import coupled_logistic_basin_grid

            rust_basin = np.asarray(
                coupled_logistic_basin_grid(
                    x_range,
                    y_range,
                    A,
                    D,
                    n_transient,
                    ref_a,
                )
            )
            np.testing.assert_array_equal(rust_basin, _golden("basin_grid"))

    @rust_extension
    def test_coupled_logistic_basin_grid_divergence_and_tie_cases(self):
        from dynachaos._rust import coupled_logistic_basin_grid

        ref_a = np.array([[0.0, 0.0]], dtype=np.float64)

        diverged = np.asarray(
            coupled_logistic_basin_grid(
                np.array([1.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
                200.0,
                0.1,
                1,
                ref_a,
            )
        )
        np.testing.assert_array_equal(diverged, np.array([[-1]], dtype=np.int8))

        tied = np.asarray(
            coupled_logistic_basin_grid(
                np.array([0.0], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                1.0,
                0.1,
                0,
                ref_a,
            )
        )
        np.testing.assert_array_equal(tied, np.array([[0]], dtype=np.int8))

    def test_compute_basins_parity(self, tmp_path):
        import importlib

        basin_mod = importlib.import_module("dynachaos.maps.coupled_logistic")

        kwargs = {
            "n_grid": 5,
            "n_transient": 6,
            "reference_transient": 8,
            "period": 4,
        }
        old_flag = basin_mod._RUST_AVAILABLE
        try:
            basin_mod._RUST_AVAILABLE = False
            python_payload = basin_mod.compute_basins(
                **kwargs,
                output_path=tmp_path / "basins_python.npz",
            )
        finally:
            basin_mod._RUST_AVAILABLE = old_flag

        expected_keys = {"x", "y", "basin", "A", "D"}
        assert set(python_payload) == expected_keys
        for key in python_payload:
            expected = _golden(f"compute_basins_{key}")
            if key == "basin":
                np.testing.assert_array_equal(python_payload[key], expected)
            else:
                np.testing.assert_allclose(python_payload[key], expected)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            try:
                basin_mod._RUST_AVAILABLE = True
                rust_payload = basin_mod.compute_basins(
                    **kwargs,
                    output_path=tmp_path / "basins_rust.npz",
                )
            finally:
                basin_mod._RUST_AVAILABLE = old_flag

            assert set(rust_payload) == expected_keys
            for key in rust_payload:
                expected = _golden(f"compute_basins_{key}")
                if key == "basin":
                    np.testing.assert_array_equal(rust_payload[key], expected)
                else:
                    np.testing.assert_allclose(rust_payload[key], expected)


class TestIntermittencyOracleParity:
    """Verify Rust intermittency oracle kernels match Python fallbacks."""

    def test_direct_pm_type_i_parity(self):
        from dynachaos.maps.intermittency import _pm_type_i_oracle_python

        kwargs = dict(n=80, x0=0.012, eps=2e-4, a=0.8, modulo=True)
        python_result = _pm_type_i_oracle_python(**kwargs)

        _assert_golden("pm_type_i_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import pm_type_i_oracle

            _assert_golden("pm_type_i_direct", pm_type_i_oracle(**kwargs))

    def test_direct_pm_type_ii_parity(self):
        from dynachaos.maps.intermittency import _pm_type_ii_oracle_python

        kwargs = dict(n=80, x0=1e-3, y0=2e-3, eps=1e-3, a=-1.0, theta=0.37)
        python_result = _pm_type_ii_oracle_python(**kwargs)

        _assert_golden("pm_type_ii_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import pm_type_ii_oracle

            _assert_golden("pm_type_ii_direct", pm_type_ii_oracle(**kwargs))

    def test_direct_pm_type_iii_parity(self):
        from dynachaos.maps.intermittency import _pm_type_iii_oracle_python

        kwargs = dict(n=80, x0=1e-3, eps=1e-3, a=1.0)
        python_result = _pm_type_iii_oracle_python(**kwargs)

        _assert_golden("pm_type_iii_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import pm_type_iii_oracle

            _assert_golden("pm_type_iii_direct", pm_type_iii_oracle(**kwargs))

    def test_direct_on_off_parity(self):
        from dynachaos.maps.intermittency import _on_off_oracle_python

        driver = np.random.default_rng(2026).normal(size=80)
        python_result = _on_off_oracle_python(driver, 1e-6, 0.0, 0.25)

        _assert_golden("on_off_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import on_off_oracle

            _assert_golden("on_off_direct", on_off_oracle(driver, 1e-6, 0.0, 0.25))

    def test_direct_on_off_skew_logistic_parity(self):
        from dynachaos.maps.intermittency import _on_off_skew_logistic_oracle_python

        kwargs = dict(n=80, x0=0.217, y0=1e-2, eps=0.499)
        python_result = _on_off_skew_logistic_oracle_python(**kwargs)

        _assert_golden("on_off_skew_logistic_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import on_off_skew_logistic_oracle

            _assert_golden("on_off_skew_logistic_direct", on_off_skew_logistic_oracle(**kwargs))

    def test_on_off_skew_logistic_public_dispatch_and_bursting(self):
        from dynachaos.maps import ON_OFF_SKEW_LOGISTIC_ONSET, on_off_skew_logistic_oracle

        np.testing.assert_allclose(ON_OFF_SKEW_LOGISTIC_ONSET, 0.5)

        out = on_off_skew_logistic_oracle(n=4_000, x0=0.217, y0=1e-2, eps=0.499)

        np.testing.assert_equal(out.shape, (4_000, 2))
        np.testing.assert_allclose(out[:, 0], np.clip(out[:, 0], 0.0, 1.0))
        y_abs = np.abs(out[:, 1])
        median = np.quantile(y_abs, 0.50)
        q95 = np.quantile(y_abs, 0.95)
        np.testing.assert_array_less(10.0 * median, q95)
        np.testing.assert_array_less(1e-3, np.max(y_abs))

    def test_direct_logistic_type_i_parity(self):
        from dynachaos.maps.intermittency import (
            LOGISTIC_TYPE_I_ONSET,
            _logistic_type_i_oracle_python,
        )

        kwargs = dict(n=80, x0=0.2, r=LOGISTIC_TYPE_I_ONSET - 1e-4)
        python_result = _logistic_type_i_oracle_python(**kwargs)

        _assert_golden("logistic_type_i_direct", python_result)

        if _RUST_IMPORTABLE and not _NO_RUST_ENV:
            from dynachaos._rust import logistic_type_i_oracle

            _assert_golden("logistic_type_i_direct", logistic_type_i_oracle(**kwargs))

    def test_public_dispatch_parity(self):
        from dynachaos.maps import intermittency as int_mod

        calls = [
            ("dispatch_pm_type_i", int_mod.pm_type_i_oracle, dict(n=50, x0=0.01, eps=1e-4, a=1.0)),
            (
                "dispatch_pm_type_ii",
                int_mod.pm_type_ii_oracle,
                dict(n=50, x0=1e-3, y0=2e-3, eps=1e-3, a=-1.0),
            ),
            (
                "dispatch_pm_type_iii",
                int_mod.pm_type_iii_oracle,
                dict(n=50, x0=1e-3, eps=1e-3, a=1.0),
            ),
            ("dispatch_logistic_type_i", int_mod.logistic_type_i_oracle, dict(n=50, x0=0.2)),
            ("dispatch_on_off", int_mod.on_off_oracle, dict(n=50, seed=2027)),
            (
                "dispatch_on_off_skew_logistic",
                int_mod.on_off_skew_logistic_oracle,
                dict(n=50, x0=0.217, y0=1e-2, eps=0.499),
            ),
        ]

        old_flag = int_mod._RUST_AVAILABLE
        try:
            for golden_name, func, kwargs in calls:
                int_mod._RUST_AVAILABLE = False
                python_result = func(**kwargs)
                _assert_golden(golden_name, python_result)

                if _RUST_IMPORTABLE and not _NO_RUST_ENV:
                    int_mod._RUST_AVAILABLE = True
                    rust_result = func(**kwargs)
                    _assert_golden(golden_name, rust_result)
        finally:
            int_mod._RUST_AVAILABLE = old_flag


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
