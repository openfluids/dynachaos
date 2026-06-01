import numpy as np
import pytest

from dynachaos.diagnostics.correlation import fit_power_law_loglog
from dynachaos.diagnostics.intermittency import (
    LaminarLengthDistribution,
    detect_laminar_phases,
    laminar_length_distribution,
)
from dynachaos.maps.intermittency import logistic_type_i_oracle


def test_recurrence_laminar_detection_reuses_vertical_lengths():
    signal = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    mask, lengths = detect_laminar_phases(signal, method="recurrence", eps=0.0, v_min=3)

    np.testing.assert_array_equal(lengths, np.array([3, 3, 3]))
    np.testing.assert_array_equal(mask, np.array([False, False, True, True, True]))


def test_period_laminar_detection_estimates_period_and_threshold():
    signal = np.array([0.0, 1.0, 0.0, 1.02, 0.01, 1.01, 0.6, -0.4])

    mask, lengths = detect_laminar_phases(signal, method="period", percentile=50)

    assert mask.dtype == np.bool_
    assert mask.shape == signal.shape
    assert lengths.size > 0
    assert np.all(lengths >= 1)


def test_variance_laminar_detection_estimates_window_and_threshold():
    signal = np.r_[np.ones(8), np.linspace(-1.0, 1.0, 8), np.ones(8)]

    mask, lengths = detect_laminar_phases(signal, method="variance", percentile=40)

    assert mask.dtype == np.bool_
    assert mask.shape == signal.shape
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

    assert mask.shape == signal.shape
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
