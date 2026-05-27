"""Tests for multifractal diagnostics."""

import os

import numpy as np
import pytest

from dynachaos.diagnostics.multifractal import local_multifractality, multifractal_spectrum

try:
    from dynachaos._rust import multifractal_moments as _mf_moments  # noqa: F401

    _HAS_RUST = not os.environ.get("DYNACHAOS_NO_RUST")
except ImportError:
    _HAS_RUST = False

needs_rust = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")


def test_uniform_field_is_monofractal():
    field = np.ones((64, 64), dtype=np.float64)
    q = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    box = np.array([2, 4, 8, 16, 32], dtype=np.int64)

    out = multifractal_spectrum(field, box_sizes=box, q_values=q)

    dq = np.asarray(out["Dq"], dtype=np.float64)
    finite = np.isfinite(dq)
    assert finite.any()
    np.testing.assert_allclose(dq[finite], 2.0, atol=1e-10)
    assert float(out["phi"]) < 1e-12


def test_uniform_field_with_truncated_edges_is_still_monofractal():
    field = np.ones((90, 90), dtype=np.float64)
    q = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    # Deliberately non-divisors of the field side length.
    box = np.array([8, 16, 22, 30], dtype=np.int64)

    out = multifractal_spectrum(field, box_sizes=box, q_values=q)

    dq = np.asarray(out["Dq"], dtype=np.float64)
    finite = np.isfinite(dq)
    assert finite.any()
    # With truncated edges, the exact slope follows the occupied-box scaling.
    n_boxes = (field.shape[0] // box) * (field.shape[1] // box)
    slope, _ = np.polyfit(np.log(box.astype(np.float64)), np.log(n_boxes.astype(np.float64)), 1)
    expected_d = -float(slope)
    np.testing.assert_allclose(dq[finite], expected_d, atol=1e-10)
    assert float(np.std(dq[finite])) < 1e-10


def test_local_multifractality_detects_heterogeneous_patch():
    rng = np.random.default_rng(42)
    field = np.ones((64, 64), dtype=np.float64)
    # Inject one heterogeneous tile.
    field[32:48, 32:48] = rng.lognormal(mean=0.0, sigma=1.0, size=(16, 16))

    out = local_multifractality(
        field,
        tile_size=16,
        box_sizes=np.array([2, 4, 8], dtype=np.int64),
        q_values=np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64),
    )
    phi = np.asarray(out["phi"], dtype=np.float64)

    assert np.nanmax(phi) > np.nanmin(phi)
    # The perturbed tile should be more multifractal than the median tile.
    assert phi[2, 2] > np.nanmedian(phi)


def test_local_multifractality_skips_zero_mass_tiles():
    field = np.zeros((64, 64), dtype=np.float64)
    field[32:48, 32:48] = 1.0

    out = local_multifractality(
        field,
        tile_size=16,
        box_sizes=np.array([2, 4, 8], dtype=np.int64),
        q_values=np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64),
    )
    phi = np.asarray(out["phi"], dtype=np.float64)
    delta = np.asarray(out["delta"], dtype=np.float64)

    assert np.isnan(phi[0, 0])
    assert np.isfinite(phi[2, 2])
    assert delta[0, 0] == pytest.approx(0.0)


@needs_rust
def test_multifractal_rust_python_parity(monkeypatch):
    import dynachaos.diagnostics.multifractal as mf_mod

    rng = np.random.default_rng(0)
    field = rng.random((48, 48), dtype=np.float64)
    q = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    box = np.array([2, 4, 8, 16], dtype=np.int64)

    old_flag = mf_mod._RUST_AVAILABLE

    monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", True)
    rust_out = mf_mod.multifractal_spectrum(field, box_sizes=box, q_values=q)

    monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", False)
    py_out = mf_mod.multifractal_spectrum(field, box_sizes=box, q_values=q)

    monkeypatch.setattr(mf_mod, "_RUST_AVAILABLE", old_flag)

    np.testing.assert_allclose(rust_out["tau"], py_out["tau"], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        rust_out["Dq"],
        py_out["Dq"],
        atol=1e-11,
        rtol=1e-11,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rust_out["alpha_legendre"],
        py_out["alpha_legendre"],
        atol=1e-11,
        rtol=1e-11,
        equal_nan=True,
    )
    assert float(rust_out["phi"]) == pytest.approx(float(py_out["phi"]), abs=1e-11)
