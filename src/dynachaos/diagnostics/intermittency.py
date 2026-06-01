"""Building-block diagnostics for intermittency in scalar signals.

The routines here expose laminar masks, laminar-run lengths, and empirical
distributions. They deliberately do not return a Pomeau-Manneville or on-off
type label; downstream analysis should interpret the statistics with the
assumption-dependent caveats from the literature.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from dynachaos.diagnostics.poincare import _auto_delay_from_autocorr
from dynachaos.diagnostics.recurrence import laminar_lengths as recurrence_laminar_lengths
from dynachaos.diagnostics.recurrence import recurrence_matrix


@dataclass(frozen=True)
class LaminarLengthDistribution:
    """Binned and discrete empirical laminar-length distributions."""

    bin_edges: np.ndarray
    density: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray


def detect_laminar_phases(
    x,
    *,
    method="recurrence",
    eps=None,
    period=None,
    window=None,
    percentile=5.0,
    v_min=2,
):
    """Detect laminar samples and laminar-run lengths in a scalar signal.

    Parameters are estimated from the signal when omitted: recurrence uses the
    existing recurrence-rate percentile threshold, period uses the first
    autocorrelation local minimum, and variance uses that same delay as its
    rolling window. The returned lengths are measured in samples.
    """
    series = _finite_series(x)
    if method == "recurrence":
        return _detect_laminar_recurrence(series, eps, percentile, v_min)
    if method == "period":
        return _detect_laminar_period(series, eps, period, percentile)
    if method == "variance":
        return _detect_laminar_variance(series, eps, window, percentile)
    raise ValueError("method must be one of: 'recurrence', 'period', 'variance'")


def laminar_length_distribution(lengths):
    """Return Freedman-Diaconis-binned and exact-count distributions."""
    lengths = _positive_lengths(lengths)
    if lengths.size == 0:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return LaminarLengthDistribution(
            bin_edges=empty_float,
            density=empty_float,
            values=empty_int,
            counts=empty_int,
            probabilities=empty_float,
        )

    bin_edges = np.histogram_bin_edges(lengths, bins="fd")
    density, bin_edges = np.histogram(lengths, bins=bin_edges, density=True)
    values, counts = np.unique(lengths, return_counts=True)
    probabilities = counts / np.sum(counts)
    return LaminarLengthDistribution(
        bin_edges=bin_edges.astype(np.float64),
        density=density.astype(np.float64),
        values=values.astype(np.int64),
        counts=counts.astype(np.int64),
        probabilities=probabilities.astype(np.float64),
    )


def _detect_laminar_recurrence(series, eps, percentile, v_min):
    result = recurrence_laminar_lengths(
        series[:, np.newaxis],
        eps=eps,
        percentile=percentile,
        v_min=v_min,
    )
    rmat, _ = recurrence_matrix(series[:, np.newaxis], eps=result.eps)
    mask = _vertical_structure_mask(rmat, v_min)
    return mask, result.lengths


def _detect_laminar_period(series, eps, period, percentile):
    if period is None:
        period = _auto_delay_from_autocorr(series)
    period = _positive_int(period, "period")
    if period >= series.size:
        raise ValueError("period must be shorter than x")

    diffs = np.abs(series[period:] - series[:-period])
    threshold = _threshold_from_percentile(diffs, eps, percentile)
    mask = np.zeros(series.size, dtype=bool)
    mask[period:] = diffs <= threshold
    lengths = _mask_run_lengths(mask)
    return mask, lengths


def _detect_laminar_variance(series, eps, window, percentile):
    if window is None:
        window = _auto_delay_from_autocorr(series)
    window = _positive_int(window, "window")
    if window > series.size:
        raise ValueError("window must be no longer than x")

    windows = np.lib.stride_tricks.sliding_window_view(series, window)
    local_std = np.std(windows, axis=1)
    threshold = _threshold_from_percentile(local_std, eps, percentile)
    window_mask = local_std <= threshold

    mask = np.zeros(series.size, dtype=bool)
    starts = np.nonzero(window_mask)[0]
    if starts.size:
        offsets = np.arange(window)
        mask[(starts[:, np.newaxis] + offsets).ravel()] = True
    lengths = _mask_run_lengths(mask)
    return mask, lengths


def _vertical_structure_mask(rmat, v_min):
    mask = np.zeros(rmat.shape[0], dtype=bool)
    for column in rmat.T:
        labels, n_labels = ndimage.label(column)
        if n_labels == 0:
            continue
        lengths = ndimage.sum(column, labels, index=np.arange(1, n_labels + 1))
        for label_id in np.nonzero(lengths >= v_min)[0] + 1:
            mask[labels == label_id] = True
    return mask


def _mask_run_lengths(mask):
    labels, n_labels = ndimage.label(np.asarray(mask, dtype=bool))
    if n_labels == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1)), dtype=np.int64)


def _threshold_from_percentile(values, eps, percentile):
    if eps is not None:
        threshold = float(eps)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("eps must be a finite non-negative number")
        return threshold

    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    return float(np.percentile(positive, percentile))


def _finite_series(x):
    series = np.asarray(x, dtype=np.float64).ravel()
    if series.size < 2:
        raise ValueError("x must contain at least two values")
    if not np.all(np.isfinite(series)):
        raise ValueError("x must contain only finite values")
    return series


def _positive_lengths(lengths):
    arr = np.asarray(lengths, dtype=np.int64).ravel()
    if arr.size == 0:
        return arr
    if np.any(arr < 1):
        raise ValueError("lengths must contain positive integers")
    return arr


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value_int != value or value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


__all__ = [
    "LaminarLengthDistribution",
    "detect_laminar_phases",
    "laminar_length_distribution",
]
