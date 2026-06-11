"""Streaming recurrence quantification without materializing a dense matrix."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from dynachaos.diagnostics._validation import (
    finite_trajectory,
    finite_nonnegative_scalar,
    nonnegative_int,
    positive_int,
)
from dynachaos.diagnostics.recurrence import (
    _paired_distances,
    _rqa_from_line_lengths,
    _validate_streaming_metric,
)


def _run_lengths(mask: np.ndarray, min_length: int) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask:
        if bool(value):
            current += 1
        else:
            if current >= min_length:
                lengths.append(current)
            current = 0
    if current >= min_length:
        lengths.append(current)
    return lengths


def _threshold_from_percentile(X: np.ndarray, metric: str, percentile: float) -> float:
    pdist_metric = "cityblock" if metric == "manhattan" else metric
    condensed = pdist(X, metric=pdist_metric)
    positive_dists = condensed[condensed > 0]
    if positive_dists.size == 0:
        return 0.0
    # recurrence_matrix percentiles over the squareform matrix, where every
    # positive pair appears twice; duplicate the condensed vector to match.
    return float(np.percentile(np.repeat(positive_dists, 2), percentile))


def _validate_inputs(X, eps, metric, percentile, l_min, v_min, theiler):
    X = finite_trajectory(X, name="X")
    if eps is not None:
        eps = finite_nonnegative_scalar(eps, name="eps")
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    _validate_streaming_metric(metric)
    l_min = positive_int(l_min, "l_min")
    v_min = positive_int(v_min, "v_min")
    if theiler is None:
        theiler = 0
    theiler = nonnegative_int(theiler, "theiler")
    return X, eps, percentile, l_min, v_min, theiler


def rqa_streaming_from_trajectory(
    X,
    eps=None,
    metric="euclidean",
    percentile=5,
    l_min=2,
    v_min=2,
    *,
    theiler=0,
    return_counts=False,
):
    """Compute exact RQA metrics by scanning diagonals and columns.

    The default return value is byte-compatible with
    :func:`dynachaos.diagnostics.recurrence.rqa`: a dict containing ``RR``,
    ``DET``, ``LAM``, ``L``, ``TT``, ``ENTR``, and ``Lmax``.  Set
    ``return_counts=True`` to also return the diagonal/vertical run lengths and
    the exact diagonal-length bins used for ``ENTR``.

    ``theiler=0`` preserves current dense-RQA semantics.  Positive values mask
    all pairs with ``abs(i - j) <= theiler`` before counting recurrences.
    """
    X, eps, percentile, l_min, v_min, theiler = _validate_inputs(
        X, eps, metric, percentile, l_min, v_min, theiler
    )
    if eps is None:
        eps = _threshold_from_percentile(X, metric, percentile)

    N = X.shape[0]
    recurrent_upper = 0
    diag_lens: list[int] = []
    for k in range(1, N):
        recurrent = _paired_distances(X[:-k], X[k:], metric) <= eps
        if k <= theiler:
            recurrent = np.zeros_like(recurrent, dtype=bool)
        recurrent_upper += int(np.sum(recurrent))
        diag_lens.extend(_run_lengths(recurrent, l_min))

    vert_lens: list[int] = []
    recurrent_all = 0
    rows = np.arange(N)
    for j in range(N):
        recurrent = _paired_distances(X, X[j : j + 1], metric) <= eps
        if theiler > 0:
            recurrent = np.logical_and(recurrent, np.abs(rows - j) > theiler)
        recurrent_all += int(np.sum(recurrent))
        vert_lens.extend(_run_lengths(recurrent, v_min))

    if theiler == 0:
        recurrent_all = N + 2 * recurrent_upper

    stats = _rqa_from_line_lengths(N * N, recurrent_all, recurrent_upper, diag_lens, vert_lens)
    if not return_counts:
        return stats

    if diag_lens:
        bins, counts = np.unique(np.asarray(diag_lens, dtype=int), return_counts=True)
    else:
        bins = np.asarray([], dtype=int)
        counts = np.asarray([], dtype=int)
    details = {
        "eps": float(eps),
        "theiler": int(theiler),
        "diagonal_lengths": np.asarray(diag_lens, dtype=int),
        "vertical_lengths": np.asarray(vert_lens, dtype=int),
        "entr_bins": bins,
        "entr_counts": counts,
    }
    return stats, details
