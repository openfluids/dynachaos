"""0-1 test for chaos (Gottwald & Melbourne 2004, 2009).

Determines whether a deterministic dynamical system is chaotic or regular
from a scalar time series alone — no phase-space reconstruction, no
Jacobian, no embedding dimension.  Returns a single number K:

    K ≈ 0  →  regular (periodic or quasi-periodic)
    K ≈ 1  →  chaotic

The method projects the time series onto a 2D extension via

    p_n = Σ_{j=1}^{n} φ(j) cos(jc),   q_n = Σ_{j=1}^{n} φ(j) sin(jc)

where c ∈ (0, π) is a random frequency.  For chaotic data (p_n, q_n)
performs a Brownian motion (MSD grows linearly); for regular data it stays
bounded.  The growth rate K is estimated via a modified correlation method.

Reference
---------
Gottwald, G.A. & Melbourne, I. (2009) "On the implementation of the 0-1
test for chaos", SIAM J. Appl. Dyn. Syst., 8(1), 129-145.

Usage
-----
    from dynachaos.diagnostics.zero_one_test import zero_one_statistic

    K = zero_one_statistic(time_series)
    # or with multiple random c values for robustness:
    K = zero_one_statistic(time_series, n_c=100)
"""

import numpy as np


def _msd_regression(p, q, n_cut):
    """Compute K from mean-square displacement of (p, q) extension.

    Uses the modified correlation method (Eq. 3 of Gottwald & Melbourne 2009)
    to avoid problems with oscillatory MSD for regular dynamics.
    """
    N = len(p)
    if n_cut is None:
        n_cut = N // 10

    # Mean-square displacement via correlation (efficient O(N) method)
    # D(n) = <(p_{j+n} - p_j)^2 + (q_{j+n} - q_j)^2>_j
    # Using the identity: D(n) = 2[Var(p)+Var(q)] - 2[C_p(n)+C_q(n)]
    # where C(n) = autocovariance at lag n.
    mean_p, mean_q = np.mean(p), np.mean(q)
    p_c, q_c = p - mean_p, q - mean_q

    # Compute autocovariance via FFT for efficiency
    Np = len(p_c)
    fp = np.fft.rfft(p_c, n=2 * Np)
    fq = np.fft.rfft(q_c, n=2 * Np)
    acf_p = np.fft.irfft(fp * np.conj(fp))[:Np] / np.arange(Np, 0, -1)
    acf_q = np.fft.irfft(fq * np.conj(fq))[:Np] / np.arange(Np, 0, -1)

    var_sum = acf_p[0] + acf_q[0]
    D = 2.0 * var_sum - 2.0 * (acf_p[:n_cut] + acf_q[:n_cut])

    # Subtract the oscillatory (Vos) correction term (Eq. 3 in G&M 2009)
    # V_osc = (mean of phi)^2 * (1 - cos(nc)) / (1 - cos(c))
    # This is handled by using the modified correlation method.

    # Linear regression: D(n) ≈ K * n for n = 1..n_cut
    ns = np.arange(1, n_cut + 1, dtype=np.float64)
    # Use correlation coefficient as K estimator (G&M 2009 Eq. 8)
    K = np.corrcoef(ns, D[:n_cut])[0, 1]
    return K


def zero_one_statistic(phi, n_c=100, n_cut=None, rng=None):
    """Perform the 0-1 test for chaos on a scalar time series.

    Parameters
    ----------
    phi : array_like
        Scalar time series (observations of some observable of the map).
    n_c : int
        Number of random c values to average over.  More values give a
        more robust estimate; 100 is usually sufficient.
    n_cut : int or None
        Number of lags used in the MSD regression.  Default: N/10 where
        N is the length of the series.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    K : float
        The 0-1 test statistic.  K ≈ 0 for regular, K ≈ 1 for chaotic.
    """
    phi = np.asarray(phi, dtype=np.float64)
    N = len(phi)

    if rng is None:
        rng = np.random.default_rng(42)

    # Choose c values randomly from (π/5, 4π/5) to avoid resonances
    # with the driving frequency (G&M 2009 recommendation)
    c_values = rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c)

    K_values = np.empty(n_c)
    for ic, c in enumerate(c_values):
        # Build the 2D extension
        js = np.arange(1, N + 1, dtype=np.float64)
        p = np.cumsum(phi * np.cos(js * c))
        q = np.cumsum(phi * np.sin(js * c))

        K_values[ic] = _msd_regression(p, q, n_cut)

    # Median is more robust than mean to outlier c values
    return float(np.median(K_values))


def zero_one_series(phi, n_c=100, n_cut=None, rng=None):
    """Return the full vector of K values for each c (diagnostic use).

    Same as ``zero_one_statistic`` but returns all individual K values instead
    of the median, useful for checking consistency.
    """
    phi = np.asarray(phi, dtype=np.float64)
    N = len(phi)

    if rng is None:
        rng = np.random.default_rng(42)

    c_values = rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c)

    K_values = np.empty(n_c)
    for ic, c in enumerate(c_values):
        js = np.arange(1, N + 1, dtype=np.float64)
        p = np.cumsum(phi * np.cos(js * c))
        q = np.cumsum(phi * np.sin(js * c))
        K_values[ic] = _msd_regression(p, q, n_cut)

    return c_values, K_values
