import numpy as np
import pytest
from conftest import logistic_series

from dynachaos.diagnostics._validation import (
    finite_positive_scalar,
    finite_series_1d,
    finite_trajectory,
    positive_int,
    sorted_nonnegative_radius_grid,
    square_bool_matrix,
)
from dynachaos.diagnostics.correlation import correlation_dimension, fit_power_law_loglog
from dynachaos.diagnostics.permutation import (
    complexity_entropy,
    ordinal_distribution,
    permutation_entropy,
)
from dynachaos.diagnostics.recurrence import (
    embed_time_delay,
    laminar_lengths,
    recurrence_matrix,
    rqa,
    rqa_from_trajectory,
)
from dynachaos.diagnostics.zero_one_test import zero_one_statistic


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_validation_helpers_reject_bool_integer(value):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        positive_int(value, "n")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([1.0, np.nan], "x must contain only finite values"),
        ([[1.0, 2.0], [np.inf, 3.0]], "X must contain only finite values"),
        (np.array(1.0), "X must be a non-empty 1D or 2D trajectory"),
        (np.empty((0, 2)), "X must be a non-empty 1D or 2D trajectory"),
        (0.0, "r must be a finite positive number"),
    ],
)
def test_validation_helpers_reject_non_finite_or_non_positive_values(value, message):
    with pytest.raises(ValueError, match=message):
        if message.startswith("x"):
            finite_series_1d(value, name="x")
        elif message.startswith("X"):
            finite_trajectory(value, name="X")
        else:
            finite_positive_scalar(value, name="r")


@pytest.mark.parametrize(
    ("r_values", "message"),
    [
        ([[0.0, 1.0]], "r_values must be a 1D array"),
        ([0.0, np.nan], "r_values must contain only finite values"),
        ([-0.1, 0.2], "r_values must be non-negative"),
        ([0.2, 0.1], "r_values must be sorted in ascending order"),
    ],
)
def test_validation_helpers_reject_invalid_radius_grids(r_values, message):
    with pytest.raises(ValueError, match=message):
        sorted_nonnegative_radius_grid(r_values, name="r_values")


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((0, 0), dtype=bool), "R must be a non-empty square matrix"),
        (np.ones((2, 3), dtype=bool), "R must be a non-empty square matrix"),
        (np.array([[True, False], [True, True]]), "R must be symmetric"),
    ],
)
def test_validation_helpers_reject_invalid_recurrence_matrices(matrix, message):
    with pytest.raises(ValueError, match=message):
        square_bool_matrix(matrix, name="R", symmetric=True)


def test_zero_one_regular_vs_chaotic():
    n = 5000
    t = np.arange(n, dtype=np.float64)
    regular = np.sin(2.0 * np.pi * 0.071 * t) + 0.2 * np.sin(2.0 * np.pi * 0.113 * t)
    chaotic = logistic_series(n=n, a=1.99, burn=2000)

    rng = np.random.default_rng(2026)
    k_regular = zero_one_statistic(regular, n_c=20, rng=rng)
    k_chaotic = zero_one_statistic(chaotic, n_c=20, rng=np.random.default_rng(2026))

    assert k_regular < 0.4
    assert k_chaotic > 0.6


def test_zero_one_zero_observable_is_regular():
    k = zero_one_statistic(np.zeros(100), n_c=5, rng=np.random.default_rng(2027))

    assert k == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("phi", "n_c", "n_cut", "message"),
    [
        ([1.0, 2.0], 1, None, "at least three"),
        ([1.0, np.nan, 2.0], 1, None, "finite values"),
        ([1.0, 2.0, 3.0, 4.0], 0, None, "n_c must be"),
        ([1.0, 2.0, 3.0, 4.0], 1.5, None, "n_c must be"),
        ([1.0, 2.0, 3.0, 4.0], 1, 1, "n_cut must be"),
        ([1.0, 2.0, 3.0, 4.0], 1, 5, "n_cut must be"),
    ],
)
def test_zero_one_statistic_rejects_fuzzed_invalid_inputs(phi, n_c, n_cut, message):
    with pytest.raises(ValueError, match=message):
        zero_one_statistic(phi, n_c=n_c, n_cut=n_cut)


def test_permutation_entropy_bounds():
    x = np.sin(np.linspace(0.0, 60.0, 2000))
    h = permutation_entropy(x, d=5)
    h2, c2 = complexity_entropy(x, d=5)

    assert 0.0 <= h <= 1.0
    assert 0.0 <= h2 <= 1.0
    assert np.isfinite(c2)
    assert c2 >= 0.0


@pytest.mark.parametrize(
    ("d", "tau", "message"),
    [
        (0, 1, "d must be >= 2"),
        (1, 1, "d must be >= 2"),
        (11, 1, "d must be <= 10"),
        (2, 0, "tau must be >= 1"),
        (2, 1.5, "d and tau must be positive integers"),
        (5, 1, "time series is too short"),
    ],
)
def test_ordinal_distribution_rejects_invalid_embedding_parameters(d, tau, message):
    with pytest.raises(ValueError, match=message):
        ordinal_distribution(np.arange(3.0), d=d, tau=tau)


def test_recurrence_and_rqa_sanity():
    t = np.linspace(0.0, 40.0, 600)
    traj = np.column_stack([np.sin(t), np.cos(t)])

    rmat, eps = recurrence_matrix(traj, percentile=8)
    stats = rqa(rmat, l_min=2, v_min=2)

    assert rmat.shape == (len(traj), len(traj))
    assert rmat.dtype == np.bool_
    assert eps > 0.0

    assert 0.0 <= stats["RR"] <= 1.0
    assert 0.0 <= stats["DET"] <= 1.0
    assert 0.0 <= stats["LAM"] <= 1.0
    assert stats["L"] >= 0.0
    assert stats["TT"] >= 0.0
    assert stats["ENTR"] >= 0.0
    assert stats["Lmax"] >= 0


@pytest.mark.parametrize(
    "metric",
    ["euclidean", "sqeuclidean", "cityblock", "manhattan", "chebyshev"],
)
def test_rqa_from_trajectory_matches_dense_recurrence_matrix(metric):
    t = np.linspace(0.0, 30.0, 180)
    traj = np.column_stack([np.sin(t), np.cos(1.7 * t)])

    dense_metric = "cityblock" if metric == "manhattan" else metric
    rmat, _ = recurrence_matrix(traj, percentile=8, metric=dense_metric)
    dense_stats = rqa(rmat, l_min=2, v_min=2)
    streaming_stats = rqa_from_trajectory(traj, percentile=8, metric=metric, l_min=2, v_min=2)

    assert streaming_stats.keys() == dense_stats.keys()
    for key, value in dense_stats.items():
        assert streaming_stats[key] == pytest.approx(value)


def test_rqa_from_trajectory_constant_signal_matches_dense():
    traj = np.ones((12, 2))

    rmat, eps = recurrence_matrix(traj)
    dense_stats = rqa(rmat, l_min=2, v_min=2)
    streaming_stats = rqa_from_trajectory(traj, l_min=2, v_min=2)

    assert eps == 0.0
    assert streaming_stats == dense_stats


def test_rqa_from_trajectory_matches_dense_with_explicit_eps():
    t = np.linspace(0.0, 24.0, 160)
    traj = np.column_stack([np.sin(t), np.cos(1.3 * t)])

    rmat, _ = recurrence_matrix(traj, eps=0.25, metric="euclidean")
    dense_stats = rqa(rmat, l_min=3, v_min=2)
    streaming_stats = rqa_from_trajectory(traj, eps=0.25, metric="euclidean", l_min=3, v_min=2)

    assert streaming_stats == pytest.approx(dense_stats)


def test_rqa_from_trajectory_percentile_threshold_matches_dense_squareform_multiset():
    # Interpolated percentile differs between the condensed pdist vector and
    # the squareform multiset; this fixture exposed eps 4.6 vs 5.0 pre-fix.
    traj = np.array([[15.0], [18.0], [23.0]])

    rmat, eps = recurrence_matrix(traj, eps=None, percentile=40)
    dense_stats = rqa(rmat, l_min=1, v_min=1)
    streaming_stats = rqa_from_trajectory(traj, eps=None, percentile=40, l_min=1, v_min=1)

    assert eps == pytest.approx(5.0)
    assert streaming_stats == pytest.approx(dense_stats)


def test_laminar_lengths_exposes_vertical_line_distribution_and_rqa_measures():
    traj = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])

    result = laminar_lengths(traj, eps=0.0, v_min=3)

    np.testing.assert_array_equal(result.lengths, np.array([3, 3, 3]))
    assert result.LAM == pytest.approx(9.0 / 13.0)
    assert result.TT == pytest.approx(3.0)
    assert result.eps == pytest.approx(0.0)


def test_laminar_lengths_matches_rqa_laminarity_and_trapping_time():
    t = np.linspace(0.0, 20.0, 140)
    traj = np.column_stack([np.sin(t), np.cos(1.2 * t)])

    rmat, eps = recurrence_matrix(traj, percentile=9)
    stats = rqa(rmat, v_min=2)
    result = laminar_lengths(traj, percentile=9, v_min=2)

    assert result.eps == pytest.approx(eps)
    assert result.LAM == pytest.approx(stats["LAM"])
    assert result.TT == pytest.approx(stats["TT"])


def test_rqa_from_trajectory_minimal_trajectory():
    # N==1: single point; eps auto-selects to 0.0 (no positive pairwise distances)
    X1 = np.ones((1, 2))
    rmat1, _ = recurrence_matrix(X1)
    dense_stats1 = rqa(rmat1, l_min=2, v_min=2)
    streaming_stats1 = rqa_from_trajectory(X1, l_min=2, v_min=2)

    assert dense_stats1["RR"] == pytest.approx(1.0)
    assert streaming_stats1.keys() == dense_stats1.keys()
    for key, value in dense_stats1.items():
        assert streaming_stats1[key] == pytest.approx(value)

    # N==2: two distinct points; explicit eps=0.5 excludes the off-diagonal pair
    X2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    rmat2, _ = recurrence_matrix(X2, eps=0.5)
    dense_stats2 = rqa(rmat2, l_min=2, v_min=2)
    streaming_stats2 = rqa_from_trajectory(X2, eps=0.5, l_min=2, v_min=2)

    assert streaming_stats2.keys() == dense_stats2.keys()
    for key, value2 in dense_stats2.items():
        assert streaming_stats2[key] == pytest.approx(value2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"X": []}, "non-empty"),
        ({"X": [0.0, np.nan, 1.0]}, "finite values"),
        ({"X": np.arange(5.0), "eps": -1.0}, "eps must be"),
        ({"X": np.arange(5.0), "percentile": 101.0}, "percentile must be"),
        ({"X": np.arange(5.0), "l_min": 0}, "l_min"),
        ({"X": np.arange(5.0), "v_min": False}, "v_min"),
        ({"X": np.arange(5.0), "metric": "cosine"}, "currently supports metric"),
    ],
)
def test_rqa_from_trajectory_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rqa_from_trajectory(**kwargs)


@pytest.mark.parametrize(
    ("rmat", "message"),
    [
        (np.array([], dtype=bool), "square"),
        (np.ones((0, 0), dtype=bool), "square"),
        (np.ones((2, 3), dtype=bool), "square"),
    ],
)
def test_rqa_rejects_invalid_matrix_shape(rmat, message):
    with pytest.raises(ValueError, match=message):
        rqa(rmat)


def test_rqa_rejects_non_symmetric_matrix():
    rmat = np.array(
        [
            [True, True, False],
            [False, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )

    with pytest.raises(ValueError, match="symmetric"):
        rqa(rmat)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"l_min": 0}, "l_min"),
        ({"v_min": 0}, "v_min"),
        ({"l_min": 1.5}, "l_min"),
        ({"v_min": None}, "v_min"),
        ({"l_min": True}, "l_min"),
        ({"v_min": False}, "v_min"),
    ],
)
def test_rqa_rejects_invalid_line_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rqa(np.eye(3, dtype=bool), **kwargs)


def test_recurrence_matrix_constant_signal_uses_zero_threshold():
    rmat, eps = recurrence_matrix(np.ones(5))

    assert eps == 0.0
    assert rmat.shape == (5, 5)
    assert np.all(rmat)


@pytest.mark.parametrize("bad_input", [[], [0.0, np.nan, 1.0], [[0.0, 1.0], [np.inf, 2.0]]])
def test_recurrence_matrix_rejects_empty_or_nonfinite_input(bad_input):
    with pytest.raises(ValueError, match="finite values|non-empty"):
        recurrence_matrix(bad_input, eps=1.0)


@pytest.mark.parametrize("eps", [-0.1, np.nan, np.inf])
def test_recurrence_matrix_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps must be"):
        recurrence_matrix(np.arange(5.0), eps=eps)


@pytest.mark.parametrize("percentile", [-1.0, 101.0, np.nan])
def test_recurrence_matrix_rejects_invalid_percentile(percentile):
    with pytest.raises(ValueError, match="percentile must be"):
        recurrence_matrix(np.arange(5.0), percentile=percentile)


def test_recurrence_matrix_translation_invariance():
    t = np.linspace(0.0, 10.0, 120)
    traj = np.column_stack([np.sin(t), np.cos(2.0 * t)])
    shifted = traj + np.array([10.0, -4.0])

    rmat, eps = recurrence_matrix(traj, percentile=7)
    shifted_rmat, shifted_eps = recurrence_matrix(shifted, percentile=7)

    assert shifted_eps == pytest.approx(eps)
    np.testing.assert_array_equal(shifted_rmat, rmat)


def test_recurrence_matrix_positive_scaling_invariance_with_percentile_threshold():
    t = np.linspace(0.0, 10.0, 120)
    traj = np.column_stack([np.sin(t), np.cos(2.0 * t)])

    rmat, eps = recurrence_matrix(traj, percentile=7)
    scaled_rmat, scaled_eps = recurrence_matrix(3.25 * traj, percentile=7)

    assert scaled_eps == pytest.approx(3.25 * eps)
    np.testing.assert_array_equal(scaled_rmat, rmat)


@pytest.mark.parametrize(
    ("d", "tau"),
    [
        (0, 1),
        (-1, 1),
        (2, 0),
        (2, -1),
        (1.5, 1),
        (2, None),
        (4, 2),
    ],
)
def test_embed_time_delay_rejects_fuzzed_impossible_parameters(d, tau):
    x = np.arange(5, dtype=np.float64)

    with pytest.raises(ValueError):
        embed_time_delay(x, d=d, tau=tau)


def test_embed_time_delay_fuzzed_valid_parameters_preserve_shape_and_values():
    rng = np.random.default_rng(2027)

    for _ in range(25):
        n = int(rng.integers(6, 30))
        d = int(rng.integers(1, 5))
        tau = int(rng.integers(1, 4))
        if n <= (d - 1) * tau:
            continue

        x = rng.normal(size=n)
        embedded = embed_time_delay(x, d=d, tau=tau)

        assert embedded.shape == (n - (d - 1) * tau, d)
        for j in range(d):
            np.testing.assert_allclose(embedded[:, j], x[j * tau : j * tau + embedded.shape[0]])


def test_correlation_dimension_circle():
    """A circle (D=1) should give D2 ~ 1."""
    t = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    traj = np.column_stack([np.cos(t), np.sin(t)])
    D2, _, _, _, _ = correlation_dimension(traj)
    assert 0.8 < D2 < 1.3


def test_fit_power_law_loglog_recovers_synthetic_exponent():
    x = np.logspace(-2.0, 2.0, 80)
    y = 2.5 * x**1.75

    slope, intercept, rvalue, slopes, scaling = fit_power_law_loglog(x, y, min_points=5)

    assert slope == pytest.approx(1.75)
    assert intercept == pytest.approx(np.log(2.5))
    assert rvalue == pytest.approx(1.0)
    assert np.all(np.isfinite(slopes))
    assert np.count_nonzero(scaling) >= 5


def test_correlation_dimension_uses_shared_power_law_fit():
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    traj = np.column_stack([np.cos(t), np.sin(t)])

    D2, r_values, C_values, slopes, scaling = correlation_dimension(traj, n_r=25)
    n_valid = len(traj) * (len(traj) - 1) // 2
    c_floor = 1.0 / np.sqrt(n_valid)
    fit_values = np.where(C_values > c_floor, C_values, np.nan)
    slope, _, _, helper_slopes, helper_scaling = fit_power_law_loglog(
        r_values, fit_values, min_points=3
    )

    assert D2 == pytest.approx(slope)
    np.testing.assert_allclose(slopes, helper_slopes, equal_nan=True)
    np.testing.assert_array_equal(scaling, helper_scaling)
