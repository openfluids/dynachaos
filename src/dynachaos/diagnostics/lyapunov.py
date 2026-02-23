"""Lyapunov exponent and spectrum computation via QR decomposition.

Implements the standard algorithm of Benettin et al. (1980) for computing the
full Lyapunov spectrum of discrete maps.  The approach evolves a set of
orthonormal tangent vectors alongside the trajectory, periodically
re-orthonormalising via QR decomposition to prevent collapse onto the most
unstable direction.

Usage
-----
For a 1D map with known derivative::

    lam = lyapunov_exponent_1d(f, df, x0, n_iter=100_000)

For an N-dimensional map with Jacobian::

    spectrum = lyapunov_spectrum(f, jac, x0, n_iter=100_000)

For sweeping a parameter and computing the maximal exponent::

    lams = lyapunov_sweep_1d(f, df, x0_func, params, n_iter=50_000,
                             n_transient=10_000)
"""

import numpy as np


def lyapunov_exponent_1d(f, df, x0, n_iter=100_000, n_transient=10_000):
    """Compute the Lyapunov exponent of a 1D map.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n).
    df : callable
        The derivative f'(x).
    x0 : float
        Initial condition.
    n_iter : int
        Number of iterations for accumulation (after transient).
    n_transient : int
        Number of transient iterations to discard.

    Returns
    -------
    float
        The Lyapunov exponent.
    """
    x = x0
    # Transient
    for _ in range(n_transient):
        x = f(x)

    # Accumulate
    log_sum = 0.0
    for _ in range(n_iter):
        deriv = abs(df(x))
        if deriv > 0:
            log_sum += np.log(deriv)
        else:
            log_sum += -100.0  # Effectively -inf, superstable
        x = f(x)

    return log_sum / n_iter


def lyapunov_spectrum(f, jac, x0, n_iter=100_000, n_transient=10_000,
                      reorth_interval=1):
    """Compute the full Lyapunov spectrum of an N-dimensional map.

    Uses QR decomposition (Benettin et al. 1980) to track all N Lyapunov
    exponents simultaneously.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n), where x is a 1D array of length N.
    jac : callable
        The Jacobian matrix J(x), returns (N, N) array.
    x0 : array_like
        Initial condition, shape (N,).
    n_iter : int
        Iterations for accumulation.
    n_transient : int
        Transient iterations to discard.
    reorth_interval : int
        QR re-orthonormalisation interval.  1 means every step (most
        accurate); larger values trade accuracy for speed.

    Returns
    -------
    spectrum : ndarray, shape (N,)
        Lyapunov exponents in descending order.
    """
    x = np.asarray(x0, dtype=np.float64)
    dim = len(x)

    # Transient
    for _ in range(n_transient):
        x = f(x)

    # Initialise orthonormal frame
    Q = np.eye(dim, dtype=np.float64)
    log_sums = np.zeros(dim, dtype=np.float64)
    count = 0

    for i in range(n_iter):
        J = jac(x)
        Q = J @ Q
        x = f(x)

        if (i + 1) % reorth_interval == 0:
            Q, R = np.linalg.qr(Q)
            # Accumulate log of diagonal (absolute value)
            diag = np.abs(np.diag(R))
            # Guard against zero
            diag = np.where(diag > 0, diag, 1e-300)
            log_sums += np.log(diag)
            count += 1

    spectrum = log_sums / (count * reorth_interval)
    # Sort descending
    return np.sort(spectrum)[::-1]


def lyapunov_max(f, jac, x0, n_iter=100_000, n_transient=10_000, rng=None):
    """Compute only the maximal Lyapunov exponent of an N-dimensional map.

    More efficient than computing the full spectrum when only the largest
    exponent is needed — tracks a single tangent vector.

    Parameters
    ----------
    f : callable
        The map x_{n+1} = f(x_n).
    jac : callable
        The Jacobian J(x), returns (N, N) array.
    x0 : array_like
        Initial condition.
    n_iter : int
        Iterations for accumulation.
    n_transient : int
        Transient iterations to discard.
    rng : numpy.random.Generator or None
        For reproducibility.  Defaults to ``default_rng(42)``.

    Returns
    -------
    float
        The maximal Lyapunov exponent.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x = np.asarray(x0, dtype=np.float64)
    dim = len(x)

    for _ in range(n_transient):
        x = f(x)

    # Random unit tangent vector
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)

    log_sum = 0.0
    for _ in range(n_iter):
        J = jac(x)
        v = J @ v
        norm_v = np.linalg.norm(v)
        if norm_v > 0:
            log_sum += np.log(norm_v)
            v /= norm_v
        else:
            log_sum += -100.0
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
        x = f(x)

    return log_sum / n_iter


def lyapunov_sweep_1d(f, df, x0_func, params, n_iter=50_000,
                      n_transient=10_000):
    """Sweep a parameter and compute the Lyapunov exponent at each value.

    Parameters
    ----------
    f : callable
        f(x, p) — the map, parameterised by p.
    df : callable
        df(x, p) — the derivative with respect to x.
    x0_func : callable
        x0_func(p) — initial condition as a function of the parameter.
    params : array_like
        1D array of parameter values to sweep.
    n_iter : int
        Iterations for accumulation at each parameter.
    n_transient : int
        Transient iterations at each parameter.

    Returns
    -------
    lams : ndarray
        Lyapunov exponent at each parameter value.
    """
    params = np.asarray(params)
    lams = np.empty(len(params))

    for i, p in enumerate(params):
        fp = lambda x, _p=p: f(x, _p)
        dfp = lambda x, _p=p: df(x, _p)
        x0 = x0_func(p)
        lams[i] = lyapunov_exponent_1d(fp, dfp, x0, n_iter, n_transient)

    return lams


def lyapunov_sweep_nd(f, jac, x0_func, params, n_iter=50_000,
                      n_transient=10_000, full_spectrum=False):
    """Sweep a parameter and compute Lyapunov exponent(s) for an ND map.

    Parameters
    ----------
    f : callable
        f(x, p) — the map, parameterised by p.
    jac : callable
        jac(x, p) — the Jacobian with respect to x.
    x0_func : callable
        x0_func(p) — initial condition as a function of the parameter.
    params : array_like
        1D array of parameter values.
    n_iter : int
        Iterations for accumulation.
    n_transient : int
        Transient iterations.
    full_spectrum : bool
        If True, return the full spectrum at each parameter value.

    Returns
    -------
    result : ndarray
        If full_spectrum is False: shape (len(params),) — maximal exponent.
        If full_spectrum is True: shape (len(params), N) — full spectrum.
    """
    params = np.asarray(params)

    results = []
    for p in params:
        fp = lambda x, _p=p: f(x, _p)
        jacp = lambda x, _p=p: jac(x, _p)
        x0 = x0_func(p)

        if full_spectrum:
            spec = lyapunov_spectrum(fp, jacp, x0, n_iter, n_transient)
            results.append(spec)
        else:
            lam = lyapunov_max(fp, jacp, x0, n_iter, n_transient)
            results.append(lam)

    return np.array(results)
