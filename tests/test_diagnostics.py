import numpy as np
import pytest
from conftest import logistic_series

from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.diagnostics.permutation import (
    complexity_entropy,
    ordinal_distribution,
    permutation_entropy,
)
from dynachaos.diagnostics.recurrence import embed_time_delay, recurrence_matrix, rqa
from dynachaos.diagnostics.zero_one_test import zero_one_statistic


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
