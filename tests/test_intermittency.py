import numpy as np
import pytest

from dynachaos.diagnostics.correlation import fit_power_law_loglog
from dynachaos.diagnostics.intermittency import (
    CandidateScalingLaw,
    compare_powerlaw_exponential,
    DiagnosticLogLogFit,
    fit_exponential,
    fit_power_law_loglog as fit_intermittency_power_law_loglog,
    fit_power_law_mle,
    LaminarLengthDistribution,
    MeanLaminarScaling,
    detect_laminar_phases,
    laminar_length_distribution,
    mean_laminar_scaling,
    powerlaw_gof,
)
from dynachaos.maps.intermittency import logistic_type_i_oracle, pm_type_i_oracle


def test_recurrence_laminar_detection_reuses_vertical_lengths():
    signal = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    mask, lengths = detect_laminar_phases(signal, method="recurrence", eps=0.0, v_min=3)

    np.testing.assert_array_equal(lengths, np.array([3, 3, 3]))
    np.testing.assert_array_equal(mask, np.array([False, False, True, True, True]))


def test_period_laminar_detection_estimates_period_and_threshold():
    signal = np.array([0.0, 1.0, 0.0, 1.02, 0.01, 1.01, 0.6, -0.4])

    mask, lengths = detect_laminar_phases(signal, method="period", percentile=50)

    assert mask.dtype == np.bool_
    np.testing.assert_equal(mask.shape, signal.shape)
    assert lengths.size > 0
    assert np.all(lengths >= 1)


def test_variance_laminar_detection_estimates_window_and_threshold():
    signal = np.r_[np.ones(8), np.linspace(-1.0, 1.0, 8), np.ones(8)]

    mask, lengths = detect_laminar_phases(signal, method="variance", percentile=40)

    assert mask.dtype == np.bool_
    np.testing.assert_equal(mask.shape, signal.shape)
    assert lengths.size > 0
    assert np.all(lengths >= 1)


def test_laminar_length_distribution_returns_fd_bins_and_discrete_counts():
    lengths = np.array([1, 1, 2, 4, 4, 4, 8])

    dist = laminar_length_distribution(lengths)

    assert isinstance(dist, LaminarLengthDistribution)
    assert dist.bin_edges.size >= 2
    np.testing.assert_array_equal(dist.values, np.array([1, 2, 4, 8]))
    np.testing.assert_array_equal(dist.counts, np.array([2, 1, 3, 1]))
    assert np.sum(dist.probabilities) == pytest.approx(1.0)


def test_type_i_logistic_period_lengths_have_negative_loglog_tail():
    signal = logistic_type_i_oracle(8000, x0=0.2)

    mask, lengths = detect_laminar_phases(signal, method="period", percentile=10)
    dist = laminar_length_distribution(lengths)
    slope, _, _, _, scaling = fit_power_law_loglog(dist.values, dist.probabilities, min_points=3)

    np.testing.assert_equal(mask.shape, signal.shape)
    assert lengths.size >= 20
    assert np.count_nonzero(scaling) >= 3
    assert slope < -0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": [1.0], "method": "period"}, "at least two"),
        ({"x": [1.0, np.nan], "method": "period"}, "finite"),
        ({"x": [1.0, 2.0], "method": "unknown"}, "method"),
        ({"x": [1.0, 2.0], "method": "period", "period": 2}, "shorter"),
    ],
)
def test_detect_laminar_phases_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        detect_laminar_phases(**kwargs)


def test_laminar_length_distribution_rejects_nonpositive_lengths():
    with pytest.raises(ValueError, match="positive"):
        laminar_length_distribution([1, 0, 2])


def test_fit_power_law_mle_recovers_continuous_type_i_exponent():
    rng = np.random.default_rng(2028)
    csn_alpha = 1.5
    samples = (1.0 - rng.random(20_000)) ** (-1.0 / (csn_alpha - 1.0))

    fit = fit_power_law_mle(samples, discrete=False)

    np.testing.assert_equal(fit.discrete, False)
    assert fit.alpha == pytest.approx(-1.5, abs=2.0 * fit.standard_error)
    assert fit.csn_alpha == pytest.approx(1.5, abs=2.0 * fit.standard_error)
    assert fit.x_min >= 1.0
    assert fit.n_tail > 10_000


def test_fit_power_law_mle_discrete_hurwitz_zeta_path_is_finite():
    rng = np.random.default_rng(2029)
    samples = rng.zipf(2.2, size=5000)

    fit = fit_power_law_mle(samples, discrete=True, min_tail=1000)

    np.testing.assert_equal(fit.discrete, True)
    assert fit.csn_alpha > 1.0
    assert fit.alpha < -1.0
    assert fit.n_tail >= 1000
    assert np.isfinite(fit.ks_distance)


def test_powerlaw_gof_does_not_reject_true_power_law():
    rng = np.random.default_rng(2030)
    csn_alpha = 1.5
    samples = (1.0 - rng.random(800)) ** (-1.0 / (csn_alpha - 1.0))
    fit = fit_power_law_mle(samples, discrete=False)

    gof = powerlaw_gof(samples, fit=fit, n_bootstrap=20, rng=2031)

    assert gof.fit == fit
    assert gof.n_bootstrap == 20
    assert 0.0 <= gof.p_value <= 1.0
    assert gof.p_value >= 0.1


def test_fit_exponential_recovers_known_rate():
    rng = np.random.default_rng(2032)
    samples = 1.0 + rng.exponential(scale=2.0, size=5000)

    fit = fit_exponential(samples, x_min=1.0)

    assert fit.rate == pytest.approx(0.5, rel=0.06)
    assert fit.scale == pytest.approx(2.0, rel=0.06)
    assert fit.n_tail == samples.size
    assert np.isfinite(fit.log_likelihood)


def test_vuong_comparison_selects_power_law_and_exponential_signs():
    rng = np.random.default_rng(2033)
    power_law = (1.0 - rng.random(8000)) ** (-1.0 / (1.5 - 1.0))
    exponential = 1.0 + rng.exponential(scale=1.5, size=20_000)

    power_law_cmp = compare_powerlaw_exponential(power_law, discrete=False)
    exponential_cmp = compare_powerlaw_exponential(exponential, discrete=False)

    assert power_law_cmp.z > 0.0
    assert power_law_cmp.log_likelihood_ratio > 0.0
    assert exponential_cmp.z < 0.0
    assert exponential_cmp.log_likelihood_ratio < 0.0


def test_intermittency_loglog_fit_is_marked_diagnostic_only():
    x = np.logspace(0.0, 2.0, 40)
    y = x**-1.5

    fit = fit_intermittency_power_law_loglog(x, y, min_points=5)

    assert isinstance(fit, DiagnosticLogLogFit)
    np.testing.assert_equal(fit.diagnostic_only, True)
    assert fit.slope == pytest.approx(-1.5)


def test_mean_laminar_scaling_recovers_type_i_beta_from_oracle_sweep():
    eps_values = np.logspace(-4.0, -2.0, 7)
    mean_lengths = []
    for eps in eps_values:
        orbit = pm_type_i_oracle(20_000, x0=0.0, eps=eps, a=1.0, modulo=False)
        crossing = np.argmax(orbit > 0.5)
        mean_lengths.append(crossing + 1)

    scaling = mean_laminar_scaling(eps_values, np.asarray(mean_lengths, dtype=np.float64))

    np.testing.assert_equal(isinstance(scaling, MeanLaminarScaling), True)
    np.testing.assert_allclose(scaling.beta, 0.5, atol=0.08)
    np.testing.assert_equal(scaling.loglog.diagnostic_only, True)


def test_mean_laminar_scaling_reports_contested_candidate_laws_and_rpd_alpha():
    eps_values = np.logspace(-4.0, -2.0, 8)
    mean_lengths = 2.0 * np.log(1.0 / eps_values) + 1.0

    scaling = mean_laminar_scaling(eps_values, mean_lengths, rpd_alpha=0.25)

    assert isinstance(scaling.inverse_epsilon, CandidateScalingLaw)
    assert isinstance(scaling.logarithmic, CandidateScalingLaw)
    assert scaling.inverse_epsilon.name == "eps^-1"
    assert scaling.logarithmic.name == "log(1/eps)"
    assert scaling.rpd_alpha == pytest.approx(0.25)
    assert scaling.logarithmic.residual_sum_squares < scaling.inverse_epsilon.residual_sum_squares
