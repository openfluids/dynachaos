"""Correlation integral and dimension (Grassberger-Procaccia algorithm).

Estimates the correlation dimension D_2 from trajectory points by computing
the correlation integral C(r) — the fraction of pairs within distance r —
and fitting the power-law scaling C(r) ~ r^{D_2}.

The improved implementation supports:
- Theiler window to exclude temporally correlated pairs
- Chebyshev (max-norm) or Euclidean distance
- Rust-accelerated exact all-pairs counting (O(1) memory per pair)

All thresholds are derived from data geometry ("organic"):
- R-range from attractor bounding-box diameter (ptp)
- Noise floor from Poisson statistics (1/sqrt(n_valid))
- Saturation/slope filter from finite-difference resolution (log_spacing)

Reference
---------
Grassberger, P. & Procaccia, I. (1983) "Measuring the strangeness of strange
  attractors", Physica D, 9(1-2), 189-208.
Theiler, J. (1986) "Spurious dimension from correlation algorithms applied to
  limited time-series data", Phys. Rev. A, 34(3), 2427-2432.
"""

import os
import time

import numpy as np

from dynachaos.diagnostics._validation import sorted_nonnegative_radius_grid
from dynachaos.utils.system import get_rss_mb

try:
    if os.environ.get("DYNACHAOS_NO_RUST"):
        raise ImportError("Rust disabled by DYNACHAOS_NO_RUST")
    from dynachaos._rust import correlation_counts as _correlation_counts_rs

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def _correlation_counts_python(traj, r_values, theiler_window, use_chebyshev):
    """Exact all-pairs correlation counts, pure Python.

    For small N uses scipy pdist; for large N uses random sampling.
    """
    N = len(traj)
    traj = np.asarray(traj, dtype=np.float64)
    r_values = np.asarray(r_values, dtype=np.float64)
    n_r = len(r_values)
    counts = np.zeros(n_r, dtype=np.int64)

    n_pairs_total = 0
    # Brute force with Theiler window support
    for i in range(N):
        j_start = i + theiler_window + 1
        if j_start >= N:
            continue
        diffs = traj[j_start:] - traj[i]
        if use_chebyshev:
            if traj.ndim == 1 or traj.shape[1] == 1:
                dists = np.abs(diffs).ravel()
            else:
                dists = np.max(np.abs(diffs), axis=1)
        else:
            if traj.ndim == 1 or traj.shape[1] == 1:
                dists = np.abs(diffs).ravel()
            else:
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        n_pairs_total += len(dists)
        for k in range(n_r):
            counts[k] += np.sum(dists < r_values[k])

    return counts, n_pairs_total


def _print_timing(backend, N, n_valid, n_r, elapsed):
    """Print verbose timing and memory info for correlation counting."""
    throughput = n_valid * n_r / elapsed / 1e6 if elapsed > 0 else 0.0
    print(
        f"  correlation_integral ({backend}): N={N:,}, pairs={n_valid:,}, "
        f"n_r={n_r}, time={elapsed:.3f}s, "
        f"{throughput:.1f}M pair-tests/s, RSS={get_rss_mb():.0f} MB"
    )


def _valid_pair_count(n, theiler_window):
    """Closed-form count of valid pairs with |i-j| > theiler_window."""
    n_eff = max(0, n - theiler_window - 1)
    return n_eff * (n_eff + 1) // 2


def _validate_correlation_inputs(r_values, theiler_window, norm):
    """Validate correlation-count parameters shared by Python and Rust paths."""
    try:
        theiler_int = int(theiler_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("theiler_window must be >= 0") from exc
    if theiler_int != theiler_window or theiler_int < 0:
        raise ValueError("theiler_window must be >= 0")
    if norm not in {"chebyshev", "euclidean"}:
        raise ValueError("norm must be one of: 'chebyshev', 'euclidean'")

    r_values = sorted_nonnegative_radius_grid(r_values, name="r_values")
    return r_values, theiler_int, norm == "chebyshev"


def _undefined_dimension_result():
    empty = np.array([], dtype=np.float64)
    return np.nan, empty, empty, empty, np.array([], dtype=bool)


def correlation_integral(
    traj, r_values, max_pairs=500_000, theiler_window=0, norm="chebyshev", verbose=False
):
    """Compute the correlation integral C(r) for an array of r values.

    C(r) = (2 / N(N-1)) * #{pairs (i,j) with |i-j| > w and dist < r}

    Parameters
    ----------
    traj : ndarray, shape (N, d) or (N,)
        Trajectory points.
    r_values : ndarray, shape (n_r,)
        Distance thresholds (should be sorted ascending).
    max_pairs : int
        Ignored when Rust is available (exact computation used).
        Python fallback uses all pairs for small N.
    theiler_window : int
        Minimum temporal separation |i-j| > theiler_window.
        Default 0 (all pairs). Recommended: mean period or
        autocorrelation time of the signal.
    norm : str
        'chebyshev' (default, max-norm) or 'euclidean'.
    verbose : bool
        If True, print timing and memory info after computation.

    Returns
    -------
    C : ndarray, shape (n_r,)
        Correlation integral values.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]
    r_values, theiler_window, use_chebyshev = _validate_correlation_inputs(
        r_values, theiler_window, norm
    )

    N = len(traj)
    n_valid = _valid_pair_count(N, theiler_window)

    if n_valid == 0:
        return np.zeros(len(r_values))

    if _RUST_AVAILABLE:
        t0 = time.perf_counter()
        counts = np.asarray(_correlation_counts_rs(traj, r_values, theiler_window, use_chebyshev))
        elapsed = time.perf_counter() - t0
        if verbose:
            _print_timing("Rust", N, n_valid, len(r_values), elapsed)
        return counts.astype(np.float64) / n_valid

    t0 = time.perf_counter()
    counts, _ = _correlation_counts_python(traj, r_values, theiler_window, use_chebyshev)
    elapsed = time.perf_counter() - t0
    if verbose:
        _print_timing("Python", N, n_valid, len(r_values), elapsed)
    return counts.astype(np.float64) / n_valid


def _find_scaling_region(log_r, log_C, min_points=5):
    """Find the scaling region via local slope plateau detection.

    Computes the local slope d(log C)/d(log r) at each point, then finds
    the longest contiguous region where the slope is stable (small std).

    Points where the slope falls below the finite-difference resolution
    (log_spacing) are excluded — this organically removes saturation and
    noise-floor regions without hardcoded thresholds.

    Parameters
    ----------
    log_r, log_C : ndarray
        Log-transformed r and C(r) values (already filtered for C > 0).
    min_points : int
        Minimum number of points required in the scaling region.

    Returns
    -------
    mask : ndarray of bool
        Points belonging to the best scaling region.
    slopes : ndarray
        Local slope at each point.
    """
    n = len(log_r)
    if n < min_points:
        return np.ones(n, dtype=bool), np.gradient(log_C, log_r)

    # Local slopes via centered finite differences
    slopes = np.gradient(log_C, log_r)

    # Organic slope filter: the finite-difference resolution is the mean
    # spacing between consecutive log(r) values.  Slope values below this
    # are indistinguishable from zero (saturation region) or noise.
    log_spacing = (log_r[-1] - log_r[0]) / (n - 1) if n > 1 else 0.0
    usable = slopes >= log_spacing
    usable_idx = np.where(usable)[0]

    if len(usable_idx) < min_points:
        # Fall back to all points if too few pass the filter
        usable_idx = np.arange(n)

    # Sliding window on usable points to find the most stable plateau
    n_u = len(usable_idx)
    u_slopes = slopes[usable_idx]

    win = max(min_points, n_u // 4)
    if win > n_u:
        mask = np.zeros(n, dtype=bool)
        mask[usable_idx] = True
        return mask, slopes

    best_std = np.inf
    best_start = 0
    for start in range(n_u - win + 1):
        s = u_slopes[start : start + win]
        std = np.std(s)
        if std < best_std:
            best_std = std
            best_start = start

    # Expand outward from best window while slope stays consistent.
    # Adaptive threshold: allow +/-3sigma of the best window's std, with a
    # floor at the log-spacing resolution (organic replacement for magic 0.05).
    center_median = np.median(u_slopes[best_start : best_start + win])
    threshold = max(3.0 * best_std, log_spacing)

    lo = best_start
    hi = best_start + win
    while lo > 0 and abs(u_slopes[lo - 1] - center_median) < threshold:
        lo -= 1
    while hi < n_u and abs(u_slopes[hi] - center_median) < threshold:
        hi += 1

    mask = np.zeros(n, dtype=bool)
    mask[usable_idx[lo:hi]] = True
    return mask, slopes


def correlation_dimension(
    traj, n_r=50, r_range=None, max_pairs=500_000, theiler_window=0, norm="chebyshev", verbose=False
):
    """Estimate the correlation dimension D_2 from trajectory points.

    Uses local slope plateau detection to identify the scaling region
    of C(r) ~ r^{D_2}, then fits a line in log-log space over that region.

    Parameters
    ----------
    traj : ndarray, shape (N, d)
        Trajectory points.
    n_r : int
        Number of r values to use (default 50).
    r_range : tuple of (float, float) or None
        Range of r values. Auto-estimated from data if None.
    max_pairs : int
        Maximum pairs for Python fallback sampling.
    theiler_window : int
        Minimum temporal separation for valid pairs.
    norm : str
        'chebyshev' or 'euclidean'.
    verbose : bool
        If True, print timing and memory info during computation.

    Returns
    -------
    D2 : float
        Estimated correlation dimension.
    r_values : ndarray
        The r values used.
    C_values : ndarray
        Corresponding C(r).
    local_slopes : ndarray
        Local slope d(log C)/d(log r) at each valid point.
    scaling_mask : ndarray of bool
        Boolean mask over r_values indicating the scaling region used for fit.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]
    if n_r < 1:
        raise ValueError("n_r must be >= 1")

    N = len(traj)
    if N < 2:
        return _undefined_dimension_result()

    if r_range is None:
        # Organic: attractor bounding-box diameter sets the scale
        diameter = np.max(np.ptp(traj, axis=0))
        if not np.isfinite(diameter) or diameter <= 0.0:
            return _undefined_dimension_result()
        r_min = diameter / N  # below this, pair count -> 0
        r_max = diameter  # beyond this, C(r) -> 1
        r_range = (r_min, r_max)
    elif (
        len(r_range) != 2
        or not np.all(np.isfinite(r_range))
        or r_range[0] <= 0.0
        or r_range[1] <= 0.0
        or r_range[0] > r_range[1]
    ):
        raise ValueError("r_range must be a positive ascending (min, max) pair")

    r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_r)
    C_values = correlation_integral(
        traj, r_values, max_pairs, theiler_window, norm, verbose=verbose
    )

    # Filter: keep points with positive C, exclude noise floor
    # Organic: Poisson noise floor — at C = 1/sqrt(n_valid), the count is
    # sqrt(n_valid) pairs with relative error n_valid^{-1/4}.
    n_valid = _valid_pair_count(N, theiler_window)
    c_floor = 1.0 / np.sqrt(n_valid) if n_valid > 0 else 1e-5
    valid = C_values > c_floor
    # Note: saturation ceiling (formerly 0.8) is now handled organically
    # inside _find_scaling_region via the log_spacing slope filter.

    # Full-length arrays for return (NaN where C <= 0)
    full_slopes = np.full(len(r_values), np.nan)
    full_scaling = np.zeros(len(r_values), dtype=bool)

    if np.sum(valid) < 3:
        return np.nan, r_values, C_values, full_slopes, full_scaling

    log_r = np.log(r_values[valid])
    log_C = np.log(C_values[valid])

    # Find scaling region via plateau detection
    scaling_local, slopes_local = _find_scaling_region(log_r, log_C)

    # Map local masks back to full-size arrays
    valid_idx = np.where(valid)[0]
    full_slopes[valid_idx] = slopes_local
    full_scaling[valid_idx[scaling_local]] = True

    # Fit D2 in the scaling region
    fit_r = log_r[scaling_local]
    fit_C = log_C[scaling_local]

    if len(fit_r) < 3:
        return np.nan, r_values, C_values, full_slopes, full_scaling

    coeffs = np.polyfit(fit_r, fit_C, 1)
    D2 = coeffs[0]

    return D2, r_values, C_values, full_slopes, full_scaling
