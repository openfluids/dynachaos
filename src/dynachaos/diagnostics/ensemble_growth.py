"""Finite-time growth of the RMS norm of an ensemble in linear units.

This measures ensemble scale about zero, without subtracting the ensemble mean.
For separation from a reference trajectory, pass perturbations from that reference.
The rate is not a Lyapunov exponent for stochastic, discrete data and does not
establish chaos. Independent, zero-mean noise added each turn without feedback
already gives spread proportional to the square root of the turn count.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynachaos.diagnostics._validation import finite_positive_scalar, positive_int

__all__ = ["EnsembleGrowthResult", "ensemble_growth"]


@dataclass(frozen=True)
class EnsembleGrowthResult:
    """Spread, fitted rate, run-bootstrap interval, and increment-permutation null.

    ``window`` contains the inclusive start and exclusive stop sample indices.
    ``g_interval`` is the percentile interval at ``confidence`` from
    ``bootstrap_rates``. ``null_exceedance`` is the fraction of ``null_samples``
    at or above ``G_max``, without a pseudocount. It is conditional on the
    observed increments, their exchangeability, and the fixed initial states.
    """

    times: NDArray[np.float64]
    spread: NDArray[np.float64]
    window: tuple[int, int]
    g: float
    g_interval: tuple[float, float]
    confidence: float
    bootstrap_rates: NDArray[np.float64]
    G_max: float
    initial_floor: float
    null_samples: NDArray[np.float64]
    null_exceedance: float


def _spread(ensemble: NDArray[np.float64]) -> NDArray[np.float64]:
    """RMS Euclidean norm, scaled before squaring to retain small values."""
    scale = np.max(np.abs(ensemble), axis=(0, 2))
    scaled = np.divide(
        ensemble, scale[None, :, None], out=np.zeros_like(ensemble), where=scale[None, :, None] > 0
    )
    return scale * np.sqrt(np.mean(np.sum(scaled * scaled, axis=2), axis=0))


def _rate(spread: NDArray[np.float64], weights: NDArray[np.float64], time_scale: float) -> float:
    if np.any(spread <= 0):
        raise ValueError(
            "fit spread must be positive at every point, including bootstrap draws; "
            "pass nonzero trajectories or choose another window"
        )
    log_spread = np.log(spread)
    return float(weights @ (log_spread - log_spread.mean()) / time_scale)


def ensemble_growth(
    ensemble: ArrayLike,
    *,
    initial_floor: float,
    bootstrap_seed: int,
    permutation_seed: int,
    times: ArrayLike | None = None,
    window: tuple[int, int] | None = None,
    n_bootstrap: int = 999,
    n_permutations: int = 999,
    confidence: float = 0.95,
) -> EnsembleGrowthResult:
    """Estimate finite-time growth of ensemble spread with explicit random seeds.

    Parameters
    ----------
    ensemble : array_like, shape (runs, steps, state) or (runs, steps)
        Finite real states, perturbations, or scalar error curves in linear
        units. Every axis must be nonempty. No mean subtraction is performed.
        Accepted dtypes are integers exactly representable in float64,
        float16, float32, and float64.
    initial_floor : float
        Required positive minimum for ``S(0)``. Raise if the initial spread is
        below this floor; never clamp it. To change the baseline, slice the
        input explicitly. The peak uses the whole input, independent of window.
    bootstrap_seed, permutation_seed : int
        Required nonnegative integer seeds for separate random generators.
        Identical inputs and seeds reproduce all samples bit for bit.
    times : array_like, shape (steps,), optional
        Finite real, strictly increasing sample times. Default: ``arange(steps)``.
        The rate has inverse time units. The null permutes sample increments;
        unequal time gaps need an independent justification of exchangeability.
    window : tuple of int, optional
        Fit sample indices ``(start, stop)``, stop exclusive. Default: all
        samples. At least five points with positive spread are required.
    n_bootstrap, n_permutations : int
        Positive draw counts. Resample whole runs with replacement for ``g``
        only; use a percentile interval at the requested confidence level.
    confidence : float
        Interval coverage strictly between zero and one. Default: 0.95.

    Returns
    -------
    EnsembleGrowthResult
        ``spread[t] = sqrt(mean_i(sum_j(ensemble[i,t,j]**2)))``;
        ``g`` is the ordinary least-squares slope of its natural logarithm;
        ``G_max = max(spread) / spread[0]``. All null draws are returned.

    Raises
    ------
    ValueError
        For invalid inputs, fewer than five fit points, a baseline below the
        stated floor, or zero spread in a fit (including a bootstrap draw).
    FloatingPointError
        If the calculation exceeds the finite floating-point range.

    Notes
    -----
    The null independently permutes the per-turn vector increments of each
    run, keeping vector components together and the initial state fixed, then
    reconstructs trajectories by cumulative sums. It assumes exchangeable
    additive increments with no feedback, conditional on each run's observed
    increments and endpoint. This preserves square-root growth from independent
    zero-mean noise; shuffling state levels would erase that accumulation.
    With endpoints fixed, monotone positive scalar growth has null exceedance
    one even for exponential growth. This null assesses ordering and transient
    peaks, not the magnitude of endpoint growth.
    The null is not a general test for feedback or chaos. The finite-time rate
    is not a Lyapunov exponent for stochastic, discrete data.
    """
    shape_message = (
        "pass finite real ensemble values with shape (runs, steps, state) or (runs, steps); "
        "accepted dtypes are integers exactly representable in float64, float16, float32, float64"
    )
    try:
        input_values = np.asarray(ensemble)
        dtype = input_values.dtype
        if dtype.kind not in "iu" and not (dtype.kind == "f" and dtype.itemsize <= 8):
            raise ValueError(shape_message)
        if dtype.kind in "iu":
            # Only integers outside float64's consecutive range need checking.
            large = input_values > 2**53
            if dtype.kind == "i":
                large |= input_values < -(2**53)
            if any(int(value) != float(value) for value in input_values[large]):
                raise ValueError(shape_message)
        values = np.asarray(input_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(shape_message) from exc
    if values.ndim not in (2, 3) or any(size == 0 for size in values.shape):
        raise ValueError(shape_message)
    if not np.all(np.isfinite(values)):
        raise ValueError(shape_message)
    if values.ndim == 2:
        values = values[:, :, None]
    runs, steps, _ = values.shape
    initial_floor = finite_positive_scalar(initial_floor, name="initial_floor")
    n_bootstrap = positive_int(n_bootstrap, "n_bootstrap")
    n_permutations = positive_int(n_permutations, "n_permutations")
    for seed, name in ((bootstrap_seed, "bootstrap_seed"), (permutation_seed, "permutation_seed")):
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer; pass an explicit seed")
    confidence_message = "confidence must be strictly between 0 and 1"
    try:
        confidence = float(confidence)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(confidence_message) from exc
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError(confidence_message)

    times_message = "pass times as a finite real, strictly increasing array with shape (steps,)"
    try:
        if times is not None and np.iscomplexobj(np.asarray(times)):
            raise ValueError(times_message)
        time_values = (
            np.arange(steps, dtype=np.float64)
            if times is None
            else np.asarray(times, dtype=np.float64)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(times_message) from exc
    if (
        time_values.shape != (steps,)
        or not np.all(np.isfinite(time_values))
        or np.any(np.diff(time_values) <= 0)
    ):
        raise ValueError(times_message)
    if window is None:
        window = (0, steps)
    if (
        not isinstance(window, tuple)
        or len(window) != 2
        or any(
            isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
            for index in window
        )
    ):
        raise ValueError("pass window as (start, stop) integer sample indices, stop exclusive")
    start, stop = window
    if not 0 <= start <= stop <= steps:
        raise ValueError("pass window indices with 0 <= start <= stop <= steps")
    count = stop - start
    if count < 5:
        raise ValueError(f"fit window has {count} points; minimum is 5; pass a longer window")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        spread = _spread(values)
        if spread[0] < initial_floor:
            raise ValueError(
                f"initial spread {spread[0]:.6g} is below initial_floor {initial_floor:.6g}; "
                "pass a smaller stated floor or slice to another baseline"
            )
        fit_times = time_values[start:stop] - time_values[start]
        centered = fit_times - fit_times.mean()
        time_scale = float(np.max(np.abs(centered)))
        centered = centered / time_scale
        weights = centered / (centered @ centered)
        g = _rate(spread[start:stop], weights, time_scale)
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        bootstrap_rates = np.empty(n_bootstrap)
        for draw in range(n_bootstrap):
            indices = bootstrap_rng.integers(runs, size=runs)
            bootstrap_rates[draw] = _rate(_spread(values[indices, start:stop]), weights, time_scale)
        tail = (1 - confidence) / 2
        lower, upper = np.quantile(bootstrap_rates, [tail, 1 - tail])

        G_max = float(spread.max() / spread[0])
        increments = np.diff(values, axis=1)
        permutation_rng = np.random.default_rng(permutation_seed)
        null_samples = np.empty(n_permutations)
        reconstructed = np.empty_like(values)
        reconstructed[:, 0] = values[:, 0]
        for draw in range(n_permutations):
            # A separate order for each run; vector components share that order.
            order = permutation_rng.permuted(
                np.broadcast_to(np.arange(steps - 1), (runs, steps - 1)), axis=1
            )
            shuffled = np.take_along_axis(increments, order[:, :, None], axis=1)
            reconstructed[:, 1:] = values[:, :1] + np.cumsum(shuffled, axis=1)
            # The endpoint is invariant under permutation; preserve exact peak ties.
            reconstructed[:, -1] = values[:, -1]
            null_samples[draw] = _spread(reconstructed).max() / spread[0]

    return EnsembleGrowthResult(
        times=time_values.copy(),
        spread=spread,
        window=(int(start), int(stop)),
        g=g,
        g_interval=(float(lower), float(upper)),
        confidence=float(confidence),
        bootstrap_rates=bootstrap_rates,
        G_max=G_max,
        initial_floor=initial_floor,
        null_samples=null_samples,
        null_exceedance=float(np.mean(null_samples >= G_max)),
    )
