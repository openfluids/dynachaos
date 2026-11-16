"""Analytical growth checks and the no-feedback random-walk control."""

from typing import TypedDict, Unpack, cast

import numpy as np
import pytest
from numpy.typing import ArrayLike, DTypeLike

from dynachaos.diagnostics import EnsembleGrowthResult, ensemble_growth


class EstimateOptions(TypedDict, total=False):
    initial_floor: float
    bootstrap_seed: int
    permutation_seed: int
    n_bootstrap: int
    n_permutations: int
    confidence: float
    times: ArrayLike | None
    window: tuple[int, int] | None


def estimate(values: ArrayLike, **kwargs: Unpack[EstimateOptions]) -> EnsembleGrowthResult:
    return ensemble_growth(
        values,
        initial_floor=kwargs.get("initial_floor", 1e-12),
        bootstrap_seed=kwargs.get("bootstrap_seed", 21),
        permutation_seed=kwargs.get("permutation_seed", 22),
        n_bootstrap=kwargs.get("n_bootstrap", 49),
        n_permutations=kwargs.get("n_permutations", 99),
        confidence=kwargs.get("confidence", 0.95),
        times=kwargs.get("times"),
        window=kwargs.get("window"),
    )


def test_scalar_exponential() -> None:
    times = np.arange(21, dtype=float)
    initial = np.linspace(0.001, 0.002, 10)
    values = initial[:, None] * np.exp(0.25 * times)
    result = estimate(values)
    assert isinstance(result, EnsembleGrowthResult)
    # Allow 1e-14 relative roundoff for RMS and exponential arithmetic.
    np.testing.assert_allclose(
        result.spread, np.sqrt(np.mean(initial**2)) * np.exp(0.25 * times), rtol=1e-14, atol=0
    )
    # The analytical slope tolerance is 1e-12 in inverse sample units.
    assert result.g == pytest.approx(0.25, abs=1e-12, rel=0)
    np.testing.assert_allclose(result.bootstrap_rates, 0.25, atol=1e-12, rtol=0)
    # Exponentiating over 20 steps permits 1e-10 absolute peak roundoff.
    assert result.G_max == pytest.approx(np.exp(5), abs=1e-10, rel=0)
    # All increments are positive: every null peak is the same fixed endpoint.
    # Allow 1e-14 relative arithmetic roundoff in endpoint ratios.
    np.testing.assert_allclose(result.null_samples, result.G_max, rtol=1e-14, atol=0)
    assert result.null_exceedance == pytest.approx(1, rel=0, abs=1e-14)
    print(
        f"scalar g: value={result.g:.17g}, expected={0.25:.17g}, difference={result.g - 0.25:.3e}"
    )


def test_non_normal_ring() -> None:
    times = np.arange(21, dtype=float)
    theta = 2 * np.pi * np.arange(10) / 10
    initial = 1e-3 * np.column_stack((np.cos(theta), np.sin(theta)))
    # Exact exp(A t), independently applied to the ring; no numerical integrator.
    propagator = np.zeros((len(times), 2, 2))
    propagator[:, 0, 0] = propagator[:, 1, 1] = np.exp(-0.2 * times)
    propagator[:, 0, 1] = 4 * times * np.exp(-0.2 * times)
    values = np.einsum("tjk,ik->itj", propagator, initial)
    result = estimate(values, window=(15, 21))
    expected = 1e-3 * np.exp(-0.2 * times) * np.sqrt(1 + 8 * times**2)
    # Closed-form ring spread, peak, and slope must agree within 1e-12.
    np.testing.assert_allclose(result.spread, expected, atol=1e-12, rtol=0)
    expected_peak = np.sqrt(201) / np.e
    # A tolerance below one cannot admit an incorrect integer peak index.
    assert np.argmax(result.spread) == pytest.approx(5, abs=1e-14, rel=0)
    assert result.G_max == pytest.approx(expected_peak, abs=1e-12, rel=0)
    expected_rate = np.polyfit(times[15:], np.log(expected[15:]), 1)[0]
    assert result.g == pytest.approx(expected_rate, abs=1e-12, rel=0)
    curve_error = float(np.max(np.abs(result.spread - expected)))
    print(
        f"ring spread max error: value={curve_error:.17g}, expected=0, difference={curve_error:.3e}"
    )
    print(
        f"ring peak: value={result.G_max:.17g}, expected={expected_peak:.17g}, "
        f"difference={result.G_max - expected_peak:.3e}"
    )


def test_random_walk_is_not_flagged_as_amplification() -> None:
    # With 2048 runs, RMS sampling error is about 1.6%; allow 0.05 in slope.
    times = np.arange(1, 129, dtype=float)
    values = np.cumsum(np.random.default_rng(2026).standard_normal((2048, len(times))), axis=1)
    result = estimate(values, times=times, n_bootstrap=19, n_permutations=199)
    loglog_slope = np.polyfit(np.log(times), np.log(result.spread), 1)[0]
    assert loglog_slope == pytest.approx(0.5, abs=0.05, rel=0)
    lower, upper = np.quantile(result.null_samples, [0.025, 0.975])
    assert lower <= result.G_max <= upper
    assert result.null_exceedance > 0.05
    print(
        f"random walk: log-log slope={loglog_slope:.8f} (expected=0.5, tolerance=0.05), "
        f"null fraction={result.null_exceedance:.8f}"
    )


def test_seeds_reproduce_every_field_and_do_not_couple_random_steps() -> None:
    values = np.random.default_rng(13).normal(size=(10, 8, 2))
    before = values.copy()
    first, second = estimate(values), estimate(values)
    # Exact equality checks bit-for-bit reproducibility with unchanged seeds.
    for name in vars(first):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    changed_bootstrap = estimate(values, bootstrap_seed=1)
    np.testing.assert_array_equal(first.null_samples, changed_bootstrap.null_samples)
    assert not np.array_equal(first.bootstrap_rates, changed_bootstrap.bootstrap_rates)
    changed_null = estimate(values, permutation_seed=1)
    np.testing.assert_array_equal(first.bootstrap_rates, changed_null.bootstrap_rates)
    assert not np.array_equal(first.null_samples, changed_null.null_samples)
    # Allow 1e-14 relative roundoff when checking the input remains unchanged.
    np.testing.assert_allclose(values, before, rtol=1e-14, atol=0)


def test_increment_null_and_bootstrap_against_direct_small_calculation() -> None:
    values = np.random.default_rng(5).normal(size=(4, 7, 2))
    result = estimate(values, n_bootstrap=9, n_permutations=11, confidence=0.8)
    bootstrap_rng = np.random.default_rng(21)
    rates = []
    for _ in range(9):
        sampled = values[bootstrap_rng.integers(4, size=4)]
        spread = np.sqrt(np.mean(np.sum(sampled**2, axis=2), axis=0))
        rates.append(np.polyfit(np.arange(7), np.log(spread), 1)[0])
    # Independent arithmetic paths agree within 1e-14 absolute roundoff.
    np.testing.assert_allclose(result.bootstrap_rates, rates, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        result.g_interval, np.quantile(rates, [0.1, 0.9]), atol=1e-14, rtol=0
    )

    permutation_rng = np.random.default_rng(22)
    null = []
    baseline = np.sqrt(np.mean(np.sum(values[:, 0] ** 2, axis=1)))
    for _ in range(11):
        paths = []
        for run in values:
            increments = np.diff(run, axis=0)[permutation_rng.permutation(6)]
            paths.append(np.vstack((run[0], run[0] + np.cumsum(increments, axis=0))))
        spread = np.sqrt(np.mean(np.sum(np.asarray(paths) ** 2, axis=2), axis=0))
        null.append(spread.max() / baseline)
    np.testing.assert_allclose(result.null_samples, null, atol=1e-14, rtol=0)
    assert result.null_exceedance == pytest.approx(
        float(np.mean(np.asarray(null) >= result.G_max)), abs=1e-14, rel=0
    )


def test_window_times_baseline_and_no_mean_subtraction() -> None:
    times = np.array([0, 0.3, 0.8, 1.1, 2, 3, 5, 8, 9], dtype=float)
    values = np.broadcast_to(np.exp(0.25 * times), (3, len(times))).copy()
    values[:, 0] = 100
    result = estimate(values, times=times, window=(2, 8))
    # The analytical rate allows 1e-12 roundoff in inverse time units.
    assert result.g == pytest.approx(0.25, abs=1e-12, rel=0)
    # A tolerance below one cannot admit different integer window indices.
    assert result.window == pytest.approx((2, 8), abs=1e-14, rel=0)
    # Allow 1e-14 absolute roundoff in the unchanged baseline and time samples.
    assert result.G_max == pytest.approx(1, abs=1e-14, rel=0)
    np.testing.assert_allclose(result.spread, values[0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.times, times, atol=1e-14, rtol=0)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_spread_avoids_underflow_and_overflow_from_squaring(scale: float) -> None:
    values = scale * np.exp(0.25 * np.arange(6))[None, :]
    result = estimate(values, initial_floor=scale / 2)
    # Relative tolerance retains sensitivity at both extreme state scales.
    np.testing.assert_allclose(result.spread, values[0], rtol=1e-14, atol=0)
    # The analytical rate allows 1e-12 roundoff in inverse sample units.
    assert result.g == pytest.approx(0.25, abs=1e-12, rel=0)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_short_fit_reports_count_and_minimum(count: int) -> None:
    with pytest.raises(ValueError, match=f"{count} points; minimum is 5"):
        estimate(np.ones((2, 8)), window=(1, 1 + count))


def test_short_default_window_and_exact_minimum() -> None:
    with pytest.raises(ValueError, match="4 points; minimum is 5"):
        estimate(np.ones((2, 4)))
    # A constant log spread has zero slope within 1e-14 arithmetic roundoff.
    assert estimate(np.ones((1, 5))).g == pytest.approx(0, abs=1e-14, rel=0)


@pytest.mark.parametrize(
    "values",
    [
        1,
        np.ones(8),
        np.ones((2, 5, 1, 1)),
        np.ones((0, 5)),
        np.ones((2, 0)),
        np.ones((2, 5, 0)),
        [[1], [1, 2]],
        np.full((2, 5), np.nan),
        np.full((2, 5), np.inf),
        np.ones((2, 5), dtype=complex),
    ],
)
def test_invalid_ensembles_say_what_to_pass(values: ArrayLike) -> None:
    with pytest.raises(
        ValueError, match=r"pass finite real ensemble.*shape \(runs, steps, state\)"
    ):
        estimate(values)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_floor": 0}, "initial_floor"),
        ({"initial_floor": np.nan}, "initial_floor"),
        ({"initial_floor": 2}, "below initial_floor"),
        ({"bootstrap_seed": None}, "bootstrap_seed"),
        ({"permutation_seed": -1}, "permutation_seed"),
        ({"permutation_seed": True}, "permutation_seed"),
        ({"bootstrap_seed": 1.5}, "bootstrap_seed"),
        ({"n_bootstrap": 0}, "n_bootstrap"),
        ({"n_permutations": 1.5}, "n_permutations"),
        ({"confidence": 1}, "confidence"),
        ({"times": np.ones(6)}, "pass times"),
        ({"times": np.arange(5)}, "pass times"),
        ({"times": np.arange(6) + 10j}, "pass times"),
        ({"times": [[0], [1, 2]]}, "pass times"),
        ({"times": ["bad"] * 6}, "pass times"),
        ({"times": [0, 1, 2, 3, 4, np.nan]}, "pass times"),
        ({"window": (-1, 6)}, "pass window"),
        ({"window": (0, 7)}, "pass window"),
        ({"window": (False, 6)}, "pass window"),
        ({"window": (0.0, 6)}, "pass window"),
    ],
)
def test_invalid_options(kwargs: EstimateOptions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate(np.ones((2, 6)), **kwargs)


def test_floor_boundary_and_zero_spread_policy() -> None:
    # Constant spread gives unit amplification and zero rate to 1e-14 roundoff.
    assert estimate(np.ones((2, 6)), initial_floor=1).G_max == pytest.approx(1, abs=1e-14, rel=0)
    values = np.ones((2, 6))
    values[:, 3] = 0
    with pytest.raises(ValueError, match="fit spread must be positive"):
        estimate(values)
    values[:, 3] = 1
    values[:, -1] = 0
    assert estimate(values, window=(0, 5)).g == pytest.approx(0, abs=1e-14, rel=0)
    with pytest.raises(ValueError, match="below initial_floor"):
        estimate(np.zeros((2, 6)))


def test_zero_bootstrap_draw_is_reported_instead_of_dropped() -> None:
    values = np.vstack((np.zeros(6), np.ones(6)))
    with pytest.raises(ValueError, match="including bootstrap draws"):
        estimate(values)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_time_scale_avoids_underflow_and_overflow(scale: float) -> None:
    samples = np.arange(5, dtype=float)
    values = np.exp(0.25 * samples)[None, :]
    result = estimate(values, times=scale * samples)
    expected = 0.25 / scale
    # Relative error measures slope accuracy across 400 decades of time units.
    assert result.g == pytest.approx(expected, rel=1e-14, abs=0)
    np.testing.assert_allclose(result.bootstrap_rates, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(result.g_interval, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [object, np.complex64, np.complex128, np.clongdouble, np.longdouble, str, bool],
)
def test_rejected_ensemble_dtypes(dtype: DTypeLike) -> None:
    if np.dtype(dtype).kind == "f" and np.dtype(dtype) == np.dtype(np.float64):
        pytest.skip("platform has no extended-precision float dtype")
    with pytest.raises(ValueError, match="integers.*float16.*float32.*float64"):
        estimate(np.ones((2, 5), dtype=dtype))


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1), 2**63 - 1, 2**64 - 1])
def test_inexact_integer_ensemble_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="integers.*float16.*float32.*float64"):
        estimate(np.full((2, 5), value))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int64, np.uint64, np.float16, np.float32, np.float64]
)
def test_accepted_ensemble_dtypes(dtype: DTypeLike) -> None:
    result = estimate(np.full((2, 5), 2, dtype=dtype))
    # Exact input states yield constant spread to 1e-14 arithmetic roundoff.
    np.testing.assert_allclose(result.spread, 2, rtol=0, atol=1e-14)
    assert result.g == pytest.approx(0, rel=0, abs=1e-14)


@pytest.mark.parametrize("value", [2**53, 2**53 + 2, -(2**63), 2**63])
def test_exact_large_integer_ensemble_is_accepted(value: int) -> None:
    result = estimate(np.full((2, 5), value))
    # Relative tolerance covers RMS roundoff at large, representable integers.
    np.testing.assert_allclose(result.spread, abs(value), rtol=1e-14, atol=0)


@pytest.mark.parametrize("confidence", ["bad", None, np.nan, -0.1, 0, 1, 1.1, np.inf])
def test_malformed_confidence_raises_value_error(confidence: object) -> None:
    # Deliberately malformed inputs test the runtime contract beyond static types.
    with pytest.raises(ValueError, match="confidence must be strictly between 0 and 1"):
        estimate(np.ones((2, 5)), confidence=cast(float, confidence))
