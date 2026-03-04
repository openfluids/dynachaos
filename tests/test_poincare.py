import numpy as np

from dynachaos.diagnostics.poincare import poincare_section


def _periodic_signal(n=4096, dt=0.01):
    t = np.arange(n, dtype=np.float64) * dt
    return np.sin(2.0 * np.pi * 2.0 * t), 1.0 / dt


def test_poincare_section_periodic_signal_returns_planes_and_metrics():
    signal, fs = _periodic_signal()
    out = poincare_section(signal, fs)

    assert out["crossing_times"].size > 20
    assert out["crossing_values"].size == out["crossing_times"].size
    assert out["delay"] >= 1
    assert out["section_points"].ndim == 2
    assert out["section_points"].shape[1] == 2
    assert "signal_derivative" in out["planes"]
    assert out["metrics"]["num_crossings"] == out["crossing_times"].size
    assert out["metrics"]["quality"] in {
        "highly_periodic",
        "periodic",
        "quasi_periodic",
        "chaotic",
        "indeterminate",
        "insufficient_data",
    }


def test_poincare_section_direction_modes_are_supported():
    signal, fs = _periodic_signal()
    up = poincare_section(signal, fs, direction="up")
    down = poincare_section(signal, fs, direction="down")
    both = poincare_section(signal, fs, direction="both")

    assert up["crossing_times"].size > 0
    assert down["crossing_times"].size > 0
    assert both["crossing_times"].size >= up["crossing_times"].size
    assert both["crossing_times"].size >= down["crossing_times"].size


def test_poincare_section_explicit_delay():
    """Exercise the explicit delay= code path."""
    signal, fs = _periodic_signal()
    out = poincare_section(signal, fs, delay=10)

    assert out["delay"] == 10
    assert out["crossing_times"].size > 0
    assert "signal_delay_pair" in out["planes"]
    pair = out["planes"]["signal_delay_pair"]
    assert pair.ndim == 2
    assert pair.shape[1] == 2


def test_poincare_section_handles_short_signal():
    signal = np.array([0.0, 1.0], dtype=np.float64)
    out = poincare_section(signal, fs=100.0)
    assert out["crossing_times"].size == 0
    assert out["section_points"].shape == (0, 2)
    assert out["metrics"]["quality"] == "insufficient_data"
