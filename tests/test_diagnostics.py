import numpy as np
from conftest import logistic_series

from dynachaos.diagnostics.correlation import correlation_dimension
from dynachaos.diagnostics.permutation import complexity_entropy, permutation_entropy
from dynachaos.diagnostics.recurrence import recurrence_matrix, rqa
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


def test_permutation_entropy_bounds():
    x = np.sin(np.linspace(0.0, 60.0, 2000))
    h = permutation_entropy(x, d=5)
    h2, c2 = complexity_entropy(x, d=5)

    assert 0.0 <= h <= 1.0
    assert 0.0 <= h2 <= 1.0
    assert np.isfinite(c2)
    assert c2 >= 0.0


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


def test_correlation_dimension_circle():
    """A circle (D=1) should give D2 ~ 1."""
    t = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    traj = np.column_stack([np.cos(t), np.sin(t)])
    D2, _, _, _, _ = correlation_dimension(traj)
    assert 0.8 < D2 < 1.3
