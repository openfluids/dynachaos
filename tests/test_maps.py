from pathlib import Path

import numpy as np
import pytest

from dynachaos.cml.correlation_figure import _fit_correlation_length
from dynachaos.cml.gcm_clusters import broad_positive_mask, compute_clusters, compute_collective
from dynachaos.cml.pattern_dynamics import SPACE_CASES
from dynachaos.cml.primitives import (
    cluster_labels_by_tolerance,
    cml_jacobian_subblock_logistic,
    cml_step,
    cml_step_logistic,
    cml_step_logistic_batch,
    gcm_step,
    sustained_positive_mask,
)
from dynachaos.diagnostics.compare_all_helpers import (
    load_or_compute_npz,
    sweep_pair_metric,
    sweep_scalar_metric,
)
from dynachaos.maps._iter import (
    iterate_unwrapped,
    run_animation_sweep,
    run_transient,
    sample_trajectory,
    trajectory_after_transient,
)
from dynachaos.maps.circle_map import circle_map, circle_map_derivative
from dynachaos.maps.coupled_logistic import (
    ATTRACTOR_CASES,
    PHASE_REQUIRED_KEYS,
    PHASE_SCHEMA_VERSION,
    PhaseDiagramPayload,
    coupled_logistic,
    coupled_logistic_jac,
)
from dynachaos.maps.coupled_logistic import (
    compute_attractors as compute_coupled_attractors,
)
from dynachaos.maps.coupled_logistic import (
    compute_basins as compute_coupled_basins,
)
from dynachaos.maps.coupled_logistic import (
    compute_phase_diagram as compute_coupled_phase_diagram,
)
from dynachaos.maps.delayed_logistic import (
    ATTRACTOR_LABELS_SHORT,
    LOCKING_LABELS_SHORT,
    compute_attractors as compute_delayed_attractors,
)
from dynachaos.maps.delayed_logistic import (
    compute_locking_sequence as compute_delayed_locking_sequence,
)
from dynachaos.maps.delayed_logistic import (
    compute_lyapunov_spectrum as compute_delayed_lyapunov_spectrum,
)
from dynachaos.maps.henon import henon, henon_jac
from dynachaos.maps.intermittency import (
    LOGISTIC_TYPE_I_ONSET,
    LORENZ_INTERMITTENCY_RHO,
    logistic_type_i_oracle,
    lorenz_1662_oracle,
    on_off_oracle,
    pm_type_i_oracle,
    pm_type_ii_oracle,
    pm_type_iii_oracle,
)
from dynachaos.maps.modulated_circle import longest_plateau_window, modulated_circle
from dynachaos.maps.primitives import (
    delayed_logistic,
    delayed_logistic_jac,
    logistic,
    logistic_derivative,
)
from dynachaos.maps.torus_doubling import (
    compute_map_I,
    compute_map_IV,
    compute_map_IV_lyapunov,
    map_I,
    map_I_jac,
    map_IV,
    map_IV_jac,
)


def test_circle_map_range_and_derivative():
    theta = 0.37
    out = circle_map(theta, A=0.1, D=0.25)
    deriv = circle_map_derivative(theta, A=0.1, D=0.25)

    assert 0.0 <= out < 1.0
    assert np.isfinite(deriv)


def test_coupled_logistic_shapes():
    state = np.array([0.2, -0.1], dtype=np.float64)
    out = coupled_logistic(state, A=1.2, D=0.08)
    jac = coupled_logistic_jac(state, A=1.2, D=0.08)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_intermittency_oracles_are_deterministic_and_finite():
    type_i = pm_type_i_oracle(64, x0=0.01, eps=1e-4, a=1.0)
    type_ii = pm_type_ii_oracle(64, x0=1e-3, y0=2e-3, eps=1e-3, a=-1.0)
    type_iii = pm_type_iii_oracle(64, x0=1e-3, eps=1e-3, a=1.0)
    logistic = logistic_type_i_oracle(64, x0=0.2, r=LOGISTIC_TYPE_I_ONSET - 1e-4)
    on_off_a = on_off_oracle(64, seed=123)
    on_off_b = on_off_oracle(64, seed=123)
    on_off_c = on_off_oracle(64, seed=124)

    assert type_i.shape == (64,)
    assert type_ii.shape == (64, 2)
    assert type_iii.shape == (64,)
    assert logistic.shape == (64,)
    assert on_off_a.shape == (64,)
    for series in (type_i, type_ii, type_iii, logistic, on_off_a):
        assert np.all(np.isfinite(series))
    np.testing.assert_allclose(on_off_a, on_off_b)
    assert not np.array_equal(on_off_a, on_off_c)


def test_lorenz_1662_oracle_reuses_flow_helper():
    traj = lorenz_1662_oracle(t_span=(0.0, 0.05), dt=0.01, t_transient=0.0)

    assert LORENZ_INTERMITTENCY_RHO == pytest.approx(166.2)
    assert traj.shape == (5, 3)
    assert np.all(np.isfinite(traj))


def test_coupled_logistic_preserves_diagonal():
    state = np.array([0.37, 0.37], dtype=np.float64)
    out = coupled_logistic(state, A=1.3, D=0.1)

    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], logistic(0.37, 1.3))


def test_coupled_logistic_respects_exchange_symmetry():
    state = np.array([0.2, -0.4], dtype=np.float64)
    swapped = state[::-1].copy()

    out = coupled_logistic(state, A=1.25, D=0.1)
    out_swapped = coupled_logistic(swapped, A=1.25, D=0.1)

    np.testing.assert_allclose(out_swapped, out[::-1])


def test_compute_coupled_phase_diagram_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "phase_diagram.npz"

    payload = compute_coupled_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=2,
        n_sample=3,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["asym"].shape == (2, 3)
    assert payload["lyap"].shape == (2, 3)
    assert int(payload["schema_version"][0]) == PHASE_SCHEMA_VERSION
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["A"], payload["A"])
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["asym"], payload["asym"])
        np.testing.assert_allclose(saved["lyap"], payload["lyap"])


def test_coupled_phase_payload_round_trip(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    payload = PhaseDiagramPayload(
        A=np.array([0.8, 1.0], dtype=np.float64),
        D=np.array([0.0, 0.1], dtype=np.float64),
        asym=np.zeros((2, 2), dtype=np.float64),
        lyap=np.ones((2, 2), dtype=np.float64),
    )
    np.savez_compressed(path, **payload.to_npz())

    with np.load(path, allow_pickle=False) as saved:
        loaded = PhaseDiagramPayload.from_npz(saved)

    assert set(PHASE_REQUIRED_KEYS) == set(payload.to_npz())
    np.testing.assert_allclose(loaded.A, payload.A)
    np.testing.assert_allclose(loaded.D, payload.D)
    np.testing.assert_allclose(loaded.asym, payload.asym)
    np.testing.assert_allclose(loaded.lyap, payload.lyap)
    assert loaded.schema_version == payload.schema_version


def test_coupled_phase_payload_rejects_stale_schema(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8], dtype=np.float64),
        D=np.array([0.0], dtype=np.float64),
        asym=np.zeros((1, 1), dtype=np.float64),
        lyap=np.zeros((1, 1), dtype=np.float64),
        schema_version=np.array([1], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match="stale phase diagram cache"):
            PhaseDiagramPayload.from_npz(saved)


def test_coupled_phase_payload_rejects_missing_keys(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8], dtype=np.float64),
        D=np.array([0.0], dtype=np.float64),
        asym=np.zeros((1, 1), dtype=np.float64),
        schema_version=np.array([PHASE_SCHEMA_VERSION], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(KeyError, match="missing keys: lyap"):
            PhaseDiagramPayload.from_npz(saved)


def test_coupled_phase_payload_rejects_grid_shape_mismatch(tmp_path):
    path = tmp_path / "phase_diagram.npz"
    np.savez_compressed(
        path,
        A=np.array([0.8, 1.0], dtype=np.float64),
        D=np.array([0.0, 0.1], dtype=np.float64),
        asym=np.zeros((2, 1), dtype=np.float64),
        lyap=np.zeros((2, 2), dtype=np.float64),
        schema_version=np.array([PHASE_SCHEMA_VERSION], dtype=np.int16),
    )

    with np.load(path, allow_pickle=False) as saved:
        with pytest.raises(ValueError, match="grid shape mismatch"):
            PhaseDiagramPayload.from_npz(saved)


def test_compute_coupled_phase_diagram_accepts_scalar_sweeps():
    payload = compute_coupled_phase_diagram(
        A_values=1.0,
        D_values=0.1,
        n_transient=1,
        n_sample=1,
        output_path=None,
    )

    assert payload["A"].shape == (1,)
    assert payload["D"].shape == (1,)
    assert payload["asym"].shape == (1, 1)
    assert payload["lyap"].shape == (1, 1)


def test_sec03_phase_diagram_finite_asym_stays_physical_after_divergence_mask():
    with np.load("figures/sec03_transition/phase_diagram.npz", allow_pickle=False) as data:
        asym = data["asym"]
        schema_version = int(data["schema_version"][0])

    finite_asym = asym[np.isfinite(asym)]

    assert schema_version == PHASE_SCHEMA_VERSION
    assert finite_asym.size > 0
    assert np.max(finite_asym) <= 2.3


def test_compute_coupled_attractors_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "attractors.npz"
    cases = ATTRACTOR_CASES[:2]

    payload = compute_coupled_attractors(
        cases=cases,
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["x_0"].shape == (4,)
    assert payload["y_0"].shape == (4,)
    assert payload["x_1"].shape == (4,)
    assert payload["y_1"].shape == (4,)
    assert int(payload["schema_version"][0]) == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["A_values"], payload["A_values"])
        np.testing.assert_allclose(saved["x_0"], payload["x_0"])
        np.testing.assert_allclose(saved["y_1"], payload["y_1"])


def test_compute_coupled_basins_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "basins.npz"

    payload = compute_coupled_basins(
        n_grid=4,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=output_path,
    )

    assert payload["x"].shape == (4,)
    assert payload["y"].shape == (4,)
    assert payload["basin"].shape == (4, 4)
    assert payload["basin"].dtype == np.int8
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["x"], payload["x"])
        np.testing.assert_array_equal(saved["basin"], payload["basin"])


# ---------------------------------------------------------------------------
# Logistic map primitives
# ---------------------------------------------------------------------------


def test_logistic_known_values():
    """f(0, a) = 1 for all a; f(1, 2) = 1 - 2 = -1."""
    assert logistic(0.0, 1.5) == 1.0
    assert logistic(1.0, 2.0) == -1.0


def test_logistic_vectorized():
    x = np.array([0.0, 0.5, 1.0])
    result = logistic(x, 2.0)
    expected = 1.0 - 2.0 * x * x
    np.testing.assert_allclose(result, expected)


def test_logistic_derivative_values():
    """f'(x) = -2ax."""
    assert logistic_derivative(0.0, 1.5) == 0.0
    assert logistic_derivative(1.0, 2.0) == -4.0


def test_logistic_derivative_vectorized():
    x = np.array([-1.0, 0.0, 0.5])
    result = logistic_derivative(x, 1.5)
    expected = -2.0 * 1.5 * x
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Delayed logistic map
# ---------------------------------------------------------------------------


def test_delayed_logistic_shape_and_finite():
    state = np.array([0.5, 0.3])
    out = delayed_logistic(state, A=0.3, D=1.8)
    jac = delayed_logistic_jac(state, A=0.3, D=1.8)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_delayed_logistic_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    state = np.array([0.5, 0.3])
    A, D = 0.3, 1.8
    jac = delayed_logistic_jac(state, A, D)

    eps = 1e-7
    for j in range(2):
        state_plus = state.copy()
        state_plus[j] += eps
        state_minus = state.copy()
        state_minus[j] -= eps
        fd_col = (delayed_logistic(state_plus, A, D) - delayed_logistic(state_minus, A, D)) / (
            2.0 * eps
        )
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_compute_delayed_attractors_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "attractors.npz"

    payload = compute_delayed_attractors(
        D_values=np.array([1.55, 1.65]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.55_x"].shape == (4,)
    assert payload["D_1.65_y"].shape == (4,)
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.55_x"], payload["D_1.55_x"])


def test_compute_delayed_attractors_rejects_rounded_key_collisions():
    with pytest.raises(ValueError, match="unique"):
        compute_delayed_attractors(
            D_values=np.array([1.554, 1.555]),
            n_transient=1,
            n_plot=1,
            output_path=None,
        )


def test_compute_delayed_lyapunov_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "lyapunov_vs_D.npz"

    payload = compute_delayed_lyapunov_spectrum(
        D_values=np.array([1.5, 1.6]),
        n_iter=4,
        n_transient=2,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["D"].shape == (2,)
    assert payload["spectra"].shape == (2, 2)
    assert np.all(np.isfinite(payload["spectra"]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["spectra"], payload["spectra"])


def test_compute_delayed_locking_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "locking_sequence.npz"

    payload = compute_delayed_locking_sequence(
        D_values=np.array([1.86, 1.88]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.860_x"].shape == (4,)
    assert payload["D_1.880_y"].shape == (4,)
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.860_x"], payload["D_1.860_x"])


def test_compute_delayed_locking_rejects_rounded_key_collisions():
    with pytest.raises(ValueError, match="unique"):
        compute_delayed_locking_sequence(
            D_values=np.array([1.86041, 1.86042]),
            n_transient=1,
            n_plot=1,
            output_path=None,
        )


def test_delayed_logistic_panel_labels_match_lyapunov_signs():
    with np.load("figures/sec05_oscillation/attractors.npz", allow_pickle=False) as data:
        attractor_D = data["D_values"]
    with np.load("figures/sec05_oscillation/locking_sequence.npz", allow_pickle=False) as data:
        locking_D = data["D_values"]
    with np.load("figures/sec05_oscillation/lyapunov_vs_D.npz", allow_pickle=False) as data:
        lyap_D = data["D"]
        lambda1 = data["spectra"][:, 0]

    expected = {
        "chaos": "positive",
        "early chaos": "positive",
        "periodic window": "negative",
        "torus": "zero",
    }

    for D_values, labels in (
        (attractor_D, ATTRACTOR_LABELS_SHORT),
        (locking_D, LOCKING_LABELS_SHORT),
    ):
        assert len(D_values) == len(labels)
        for D, label in zip(D_values, labels, strict=True):
            idx = np.argmin(np.abs(lyap_D - D))
            lam = lambda1[idx]
            if expected[label] == "positive":
                assert lam > 1e-3, f"{label=} at D={D} must have positive lambda1, got {lam}"
            elif expected[label] == "negative":
                assert lam < -1e-3, f"{label=} at D={D} must have negative lambda1, got {lam}"
            else:
                assert abs(lam) <= 1e-3, f"{label=} at D={D} must have lambda1 near zero, got {lam}"


def test_sali_curves_store_measured_lambda1_regime_context():
    with np.load("figures/sec11_diagnostics/sali_comparison.npz", allow_pickle=False) as data:
        DB_values = data["DB_values"]
        lambda1_values = data["lambda1_values"]

    np.testing.assert_allclose(DB_values, np.array([2.35, 2.37, 2.47, 2.55]))
    assert abs(lambda1_values[0]) <= 1e-3
    assert abs(lambda1_values[1]) <= 1e-3
    assert lambda1_values[2] > 1e-3
    assert lambda1_values[3] > 5e-2


def test_sec06_canonical_lyapunov_cache_replaces_duplicate_D2_file():
    duplicate_path = Path("figures/sec06_three_torus/lyapunov_vs_D2.npz")
    with np.load("figures/sec06_three_torus/lyapunov_vs_DB.npz", allow_pickle=False) as data:
        DB = data["DB"]
        spectra = data["eps_0.005_spectra"]

    torus_idx = np.argmin(np.abs(DB - 2.35))
    chaos_idx = np.argmin(np.abs(DB - 2.55))

    assert not duplicate_path.exists()
    assert abs(spectra[torus_idx, 0]) <= 1e-3
    assert abs(spectra[torus_idx, 1]) <= 5e-3
    assert spectra[chaos_idx, 0] > 5e-2
    assert spectra[chaos_idx, 1] > 1e-2


def test_sec09_spatial_activity_separates_pattern_phase_samples():
    with np.load("figures/sec09_pattern/phase_diagram.npz", allow_pickle=False) as data:
        a_values = data["a"]
        eps_values = data["eps"]
        spatial_activity = data["spatial_activity"]

    phase_activity = []
    for a, eps, _label, _tag in SPACE_CASES:
        ia = np.argmin(np.abs(a_values - a))
        ie = np.argmin(np.abs(eps_values - eps))
        phase_activity.append(spatial_activity[ie, ia])

    phase_activity = np.asarray(phase_activity)
    rounded_groups = np.unique(np.round(phase_activity, 1))

    assert np.all(np.isfinite(phase_activity))
    assert np.ptp(phase_activity) > 0.25
    assert rounded_groups.size >= 3


# ---------------------------------------------------------------------------
# Torus doubling maps
# ---------------------------------------------------------------------------


def test_map_I_shape():
    state = np.array([0.5, 0.3, 0.4])
    out = map_I(state, A=0.4, D=2.0)
    jac = map_I_jac(state, A=0.4, D=2.0)

    assert out.shape == (3,)
    assert jac.shape == (3, 3)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_map_IV_shape():
    state = np.array([0.5, 0.3, 0.4, 0.2])
    out = map_IV(state, A=0.3, D=1.5)
    jac = map_IV_jac(state, A=0.3, D=1.5)

    assert out.shape == (4,)
    assert jac.shape == (4, 4)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_compute_torus_map_I_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_I_attractors.npz"

    payload = compute_map_I(
        D_values=np.array([2.11, 2.16]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_2.11_traj"].shape[1] == 3
    assert payload["D_2.16_traj"].shape[1] == 3
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_2.11_traj"], payload["D_2.11_traj"])


def test_compute_torus_map_IV_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_IV_attractors.npz"

    payload = compute_map_IV(
        D_values=np.array([1.515, 1.5206]),
        n_transient=2,
        n_plot=4,
        output_path=output_path,
    )

    assert payload["D_values"].shape == (2,)
    assert payload["D_1.515_traj"].shape[1] == 4
    assert payload["D_1.5206_traj"].shape[1] == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D_values"], payload["D_values"])
        np.testing.assert_allclose(saved["D_1.515_traj"], payload["D_1.515_traj"])


def test_compute_torus_map_IV_lyapunov_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "map_IV_lyapunov.npz"

    payload = compute_map_IV_lyapunov(
        D_values=np.array([1.5, 1.51]),
        n_iter=4,
        n_transient=2,
        output_path=output_path,
        progress_interval=0,
    )

    assert payload["D"].shape == (2,)
    assert payload["spectra"].shape == (2, 4)
    assert np.all(np.isfinite(payload["spectra"]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["D"], payload["D"])
        np.testing.assert_allclose(saved["spectra"], payload["spectra"])



def test_map_IV_off_symmetry_ic_and_lyapunov_physics():
    """Physics-gated regression guard for the off-symmetry IC in Map IV.

    Two assertions:
    1. Lyapunov physics: lambda1 <= 1e-3 at torus (D~1.515) and lambda1 > 0 at
       chaos onset (D~1.5212), confirming the torus->chaos transition is correctly
       resolved with the off-symmetry IC.
    2. Off-symmetry: the trajectory at D=1.515 started from x0=[0.5,0.45,0.52,0.48]
       stays off the invariant subspace {X=Z, Y=W} after transient, i.e. max|X-Z|
       and max|Y-W| are both > 1e-6. This is the regression guard that proves the IC
       is not symmetric.
    """
    # --- Part 1: Lyapunov physics ---
    payload_lyap = compute_map_IV_lyapunov(
        D_values=np.linspace(1.48, 1.53, 200),
        n_iter=5000,
        n_transient=2000,
        output_path=None,
        progress_interval=0,
    )
    D = payload_lyap["D"]
    lambda1 = payload_lyap["spectra"][:, 0]

    idx_torus = np.argmin(np.abs(D - 1.515))
    idx_chaos = np.argmin(np.abs(D - 1.5212))

    assert lambda1[idx_torus] <= 1.0e-3, (
        f"Expected lambda1 <= 1e-3 at D~1.515 (torus), got {lambda1[idx_torus]:.6f}. "
        "Symmetric IC may still be in use."
    )
    assert lambda1[idx_chaos] > 0, (
        f"Expected lambda1 > 0 at D~1.5212 (chaos onset), got {lambda1[idx_chaos]:.6f}. "
        "Off-symmetry IC may not be exploring 4D dynamics."
    )

    # --- Part 2: Off-symmetry regression guard ---
    payload_traj = compute_map_IV(
        D_values=np.array([1.515]),
        n_transient=2000,
        n_plot=10000,
        output_path=None,
    )
    traj = payload_traj["D_1.515_traj"]  # shape (10000, 4)
    X, Y, Z, W = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    max_xz = np.max(np.abs(X - Z))
    max_yw = np.max(np.abs(Y - W))

    assert max_xz > 1e-6, (
        f"Orbit stayed on symmetric manifold: max|X-Z| = {max_xz:.2e}. "
        "IC may have collapsed back onto the X=Z, Y=W subspace."
    )
    assert max_yw > 1e-6, (
        f"Orbit stayed on symmetric manifold: max|Y-W| = {max_yw:.2e}. "
        "IC may have collapsed back onto the X=Z, Y=W subspace."
    )


def test_run_animation_sweep_returns_and_writes_payload(tmp_path):
    output_path = tmp_path / "animation.npz"

    def iterate_fn(param):
        return np.column_stack(
            (
                np.full(3, param, dtype=np.float64),
                np.arange(3, dtype=np.float64),
            )
        )

    payload = run_animation_sweep(
        iterate_fn,
        np.array([0.1, 0.2]),
        output_path,
        n_plot=3,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["param_values"], np.array([0.1, 0.2]))
    np.testing.assert_allclose(payload["all_x"], np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
    np.testing.assert_allclose(payload["all_y"], np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["all_x"], payload["all_x"])


# ---------------------------------------------------------------------------
# Modulated circle map
# ---------------------------------------------------------------------------


def test_modulated_circle_range():
    state = np.array([0.3, 0.7])
    out = modulated_circle(state, A=0.15, C=0.618, D=0.25, eps=0.05)

    assert out.shape == (2,)
    assert 0.0 <= out[0] < 1.0
    assert 0.0 <= out[1] < 1.0


def test_longest_plateau_window_selects_widest_run():
    D = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rho = np.array([0.00, 0.25, 0.2502, 0.40, 0.25, 0.2501, 0.2502])

    window = longest_plateau_window(D, rho, target=0.25, tol=5e-4)

    assert window == (0.4, 0.6)


def test_broad_positive_mask_rejects_short_spikes():
    values = np.array([-0.1, 0.03, -0.02, 0.04, 0.05, 0.06, 0.07, -0.01])

    mask = broad_positive_mask(values, threshold=0.02, min_run=3)

    np.testing.assert_array_equal(
        mask,
        np.array([False, False, False, True, True, True, True, False]),
    )


def test_sustained_positive_mask_matches_legacy_alias():
    values = np.array([-0.1, 0.03, -0.02, 0.04, 0.05, 0.06, 0.07, -0.01])

    np.testing.assert_array_equal(
        sustained_positive_mask(values, threshold=0.02, min_run=3),
        broad_positive_mask(values, threshold=0.02, min_run=3),
    )


def test_compute_clusters_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "gcm_clusters.npz"

    payload = compute_clusters(
        n_sites=8,
        n_transient=2,
        n_record=4,
        seed=123,
        output_path=output_path,
    )

    assert payload["cluster_labels"].shape == (4, 8)
    assert payload["x_record"].shape == (4, 8)
    assert int(payload["N"][0]) == 8
    assert int(payload["n_transient"][0]) == 2
    assert int(payload["n_record"][0]) == 4
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_array_equal(saved["cluster_labels"], payload["cluster_labels"])
        np.testing.assert_allclose(saved["x_record"], payload["x_record"])


def test_compute_collective_returns_and_writes_explicit_payload(tmp_path):
    output_path = tmp_path / "collective_lyapunov.npz"
    a_values = np.array([1.4, 1.6])

    payload = compute_collective(
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=4,
        renorm_interval=2,
        seed=123,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["a_values"], a_values)
    assert payload["lyap_c"].shape == (2,)
    assert np.all(np.isfinite(payload["lyap_c"]))
    assert int(payload["N"][0]) == 8
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(payload)
        np.testing.assert_allclose(saved["a_values"], payload["a_values"])
        np.testing.assert_allclose(saved["lyap_c"], payload["lyap_c"])


def test_cluster_labels_by_tolerance_groups_sorted_runs():
    values = np.array([0.10, 0.1000002, 0.50, 0.5000001, 0.90])

    labels = cluster_labels_by_tolerance(values, tol=1e-5)

    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 2]))


def test_iterate_unwrapped_scalar_matches_manual_accumulation():
    value = 0.1
    for _ in range(5):
        value += 0.25 + 0.1 * value

    helper = iterate_unwrapped(0.1, lambda x: 0.25 + 0.1 * x, 5)

    np.testing.assert_allclose(helper, value)


def test_run_transient_and_sample_helpers_record_post_step_states():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    after = run_transient(state, step_fn, 2)
    np.testing.assert_allclose(after, np.array([2.0, 3.0]))

    samples = sample_trajectory(after, step_fn, 3)
    np.testing.assert_allclose(
        samples,
        np.array([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]),
    )

    combined = trajectory_after_transient(state, step_fn, 2, 3)
    np.testing.assert_allclose(combined, samples)


def test_cml_step_matches_manual_generic_update():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    eps = 0.2

    def f(arr):
        return 2.0 * arr

    def g(arr):
        return arr + 1.0

    out = cml_step(x, f, g, eps)
    manual = f(x) + eps / 2.0 * (np.roll(g(x), -1) + np.roll(g(x), 1) - 2.0 * g(x))

    np.testing.assert_allclose(out, manual)


def test_cml_step_default_preserves_flattened_roll_behavior_for_2d_input():
    x = np.array([[0.1, -0.2], [0.3, -0.4]])
    eps = 0.2

    def f(arr):
        return 2.0 * arr

    def g(arr):
        return arr + 1.0

    out = cml_step(x, f, g, eps)
    gx = g(x)
    manual = f(x) + eps / 2.0 * (np.roll(gx, -1) + np.roll(gx, 1) - 2.0 * gx)

    np.testing.assert_allclose(out, manual)


def test_cml_step_logistic_axis_argument_enables_rowwise_topology():
    x = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.5]])
    eps = 0.2

    explicit = cml_step_logistic(x, 1.6, eps, axis=1)
    rowwise = np.vstack([cml_step_logistic(row, 1.6, eps) for row in x])

    np.testing.assert_allclose(explicit, rowwise)


def test_cml_step_logistic_batch_matches_rowwise_update():
    x = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.5]])
    a_col = np.array([[1.5], [1.8]])
    eps = 0.2

    batch = cml_step_logistic_batch(x, a_col, eps)
    rowwise = np.vstack(
        [cml_step_logistic(x[idx], float(a_col[idx, 0]), eps) for idx in range(len(x))]
    )

    np.testing.assert_allclose(batch, rowwise)


def test_cml_jacobian_subblock_logistic_matches_manual_matrix():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    a = 1.5
    eps = 0.2
    L = 3

    J = cml_jacobian_subblock_logistic(x, a, eps, L)
    dfx = logistic_derivative(x, a)
    expected = np.array(
        [
            [(1.0 - eps) * dfx[0], (eps / 2.0) * dfx[1], 0.0],
            [(eps / 2.0) * dfx[0], (1.0 - eps) * dfx[1], (eps / 2.0) * dfx[2]],
            [0.0, (eps / 2.0) * dfx[1], (1.0 - eps) * dfx[2]],
        ]
    )

    np.testing.assert_allclose(J, expected)


def test_correlation_length_fit_uses_decay_head_not_noise_floor():
    r = np.arange(0, 80)
    corr = np.exp(-r / 2.0)
    corr[6:] = 2e-3 * (1.0 + 0.1 * np.sin(r[6:]))

    xi = _fit_correlation_length(r, corr)

    assert 1.0 < xi < 3.0


def test_correlation_length_fit_ignores_late_tail_spikes():
    r = np.arange(0, 80)
    corr = np.exp(-r / 0.8)
    corr[10] = 0.5

    xi = _fit_correlation_length(r, corr)

    assert 0.5 < xi < 1.0


def test_sec08_correlation_lengths_are_physical_after_refit():
    with np.load("figures/sec08_sti/correlation_decay.npz", allow_pickle=False) as data:
        a_corr = data["a_corr"]
        xi_values = data["xi_values"]

    xi_by_a = dict(zip(a_corr, xi_values, strict=True))

    assert np.all(np.isfinite(xi_values))
    assert np.all((0.0 < xi_values) & (xi_values < 10.0))
    assert xi_by_a[1.85] < xi_by_a[1.7]
    assert xi_by_a[1.95] < xi_by_a[1.7]


def test_gcm_step_matches_manual_formula():
    x = np.array([0.1, -0.2, 0.3, -0.4])
    a = 1.7
    eps = 0.15

    out = gcm_step(x, a, eps)
    fx = logistic(x, a)
    manual = (1.0 - eps) * fx + eps * np.mean(fx)

    np.testing.assert_allclose(out, manual)


def test_run_transient_returns_none_when_diverged():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    def diverged_fn(s):
        return s[0] > 2.5

    out = run_transient(state, step_fn, 5, diverged_fn=diverged_fn)
    assert out is None


def test_sample_trajectory_allow_partial_returns_prefix():
    state = np.array([0.0, 1.0])

    def step_fn(s):
        return s + 1.0

    def diverged_fn(s):
        return s[0] > 2.5

    samples = sample_trajectory(
        state,
        step_fn,
        5,
        diverged_fn=diverged_fn,
        allow_partial=True,
    )

    np.testing.assert_allclose(samples, np.array([[1.0, 2.0], [2.0, 3.0]]))


def test_iterate_unwrapped_vectorized_updates_elementwise():
    state = np.array([0.0, 1.0, 2.0])
    out = iterate_unwrapped(state, lambda s: np.array([1.0, -1.0, 0.5]), 3)
    np.testing.assert_allclose(out, np.array([3.0, -2.0, 3.5]))


def test_sweep_metric_helpers_return_expected_arrays():
    values = np.array([1.0, 2.0, 3.0])

    def series_fn(v):
        return np.array([v, 2.0 * v])

    def scalar_metric(s):
        return float(np.sum(s))

    def pair_metric(s):
        return float(np.min(s)), float(np.max(s))

    scalar = sweep_scalar_metric(values, series_fn, scalar_metric)
    first, second = sweep_pair_metric(values, series_fn, pair_metric)

    np.testing.assert_allclose(scalar, np.array([3.0, 6.0, 9.0]))
    np.testing.assert_allclose(first, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(second, np.array([2.0, 4.0, 6.0]))


def test_load_or_compute_npz_computes_when_missing(tmp_path):
    path = tmp_path / "sample.npz"

    def compute_fn():
        np.savez_compressed(path, values=np.array([1.0, 2.0]))

    data = load_or_compute_npz(path, "sample", compute_fn)

    np.testing.assert_allclose(data["values"], np.array([1.0, 2.0]))


def test_load_or_compute_npz_recomputes_when_required_keys_missing(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez_compressed(path, stale=np.array([0.0]))
    calls = 0

    def compute_fn():
        nonlocal calls
        calls += 1
        np.savez_compressed(path, values=np.array([3.0, 4.0]))

    data = load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))

    assert calls == 1
    np.testing.assert_allclose(data["values"], np.array([3.0, 4.0]))


def test_load_or_compute_npz_raises_when_compute_leaves_required_keys_missing(tmp_path):
    path = tmp_path / "sample.npz"

    def compute_fn():
        np.savez_compressed(path, other=np.array([1.0]))

    with pytest.raises(KeyError, match="missing required keys"):
        load_or_compute_npz(path, "sample", compute_fn, required_keys=("values",))


# ---------------------------------------------------------------------------
# Henon map
# ---------------------------------------------------------------------------


def test_henon_shape_and_finite():
    """Output shapes and finiteness."""
    state = np.array([0.1, 0.1])
    out = henon(state)
    jac = henon_jac(state)

    assert out.shape == (2,)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(jac))


def test_henon_known_value():
    """At origin: x' = 1, y' = 0."""
    out = henon(np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_henon_jac_finite_difference():
    """Jacobian should match finite-difference approximation."""
    state = np.array([0.63, -0.19])
    a, b = 1.4, 0.3
    jac = henon_jac(state, a, b)

    eps = 1e-7
    for j in range(2):
        state_plus = state.copy()
        state_plus[j] += eps
        state_minus = state.copy()
        state_minus[j] -= eps
        fd_col = (henon(state_plus, a, b) - henon(state_minus, a, b)) / (2.0 * eps)
        np.testing.assert_allclose(jac[:, j], fd_col, atol=1e-5)


def test_compute_clusters_seed_controls_rng_determinism():
    first = compute_clusters(seed=7, n_sites=8, n_transient=2, n_record=4, output_path=None)
    second = compute_clusters(seed=7, n_sites=8, n_transient=2, n_record=4, output_path=None)

    np.testing.assert_array_equal(first["cluster_labels"], second["cluster_labels"])
    np.testing.assert_array_equal(first["x_record"], second["x_record"])

    different_seed = compute_clusters(seed=99, n_sites=8, n_transient=2, n_record=4, output_path=None)

    assert not np.array_equal(
        first["cluster_labels"], different_seed["cluster_labels"]
    ) or not np.array_equal(
        first["x_record"], different_seed["x_record"]
    ), "compute_clusters should produce different stochastic payloads for different seeds"


def test_compute_collective_seed_controls_rng_determinism():
    a_values = np.array([1.4, 1.6, 1.8])

    first = compute_collective(
        seed=7,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )
    second = compute_collective(
        seed=7,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )

    np.testing.assert_array_equal(first["lyap_c"], second["lyap_c"])

    different_seed = compute_collective(
        seed=99,
        n_sites=8,
        a_values=a_values,
        n_transient=2,
        n_measure=6,
        renorm_interval=2,
        output_path=None,
        progress_interval=0,
    )

    assert not np.array_equal(
        first["lyap_c"], different_seed["lyap_c"]
    ), "compute_collective should produce different lyap_c values for different seeds"


# ---------------------------------------------------------------------------
# Golden-value regression tests for compute-aggregate functions
# ---------------------------------------------------------------------------


# params: D_values=[1.5, 1.6], n_iter=20, n_transient=5, seed=N/A
def test_compute_delayed_lyapunov_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "lyapunov_golden.npz"
    payload = compute_delayed_lyapunov_spectrum(
        D_values=np.array([1.5, 1.6]),
        n_iter=20,
        n_transient=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["D"], np.array([1.5, 1.6]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["spectra"],
        np.array(
            [
                [0.075066870334274, 0.07028722946827429],
                [0.0970755525312533, 0.09294026515025561],
            ]
        ),
        rtol=1e-9,
    )
    assert payload["spectra"].shape == (2, 2)
    assert np.all(np.isfinite(payload["spectra"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["spectra"], payload["spectra"], rtol=1e-9)


# params: D_values=[1.5, 1.51], n_iter=20, n_transient=5, seed=N/A
def test_compute_map_IV_lyapunov_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "map_iv_lyap_golden.npz"
    payload = compute_map_IV_lyapunov(
        D_values=np.array([1.5, 1.51]),
        n_iter=20,
        n_transient=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["D"], np.array([1.5, 1.51]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["spectra"],
        np.array(
            [
                [
                    0.1523415952594922,
                    0.03859679338679587,
                    0.014645886790399288,
                    -0.10933648083443089,
                ],
                [
                    0.15281204945237534,
                    0.04010023664586028,
                    0.012345832346427799,
                    -0.11445889859259217,
                ],
            ]
        ),
        rtol=1e-9,
    )
    assert payload["spectra"].shape == (2, 4)
    assert np.all(np.isfinite(payload["spectra"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["spectra"], payload["spectra"], rtol=1e-9)


# params: a=1.55, eps=0.1, n_sites=4, n_transient=5, n_record=3, seed=7
def test_compute_clusters_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "gcm_clusters_golden.npz"
    payload = compute_clusters(
        a=1.55,
        eps=0.1,
        n_sites=4,
        n_transient=5,
        n_record=3,
        seed=7,
        output_path=output_path,
    )

    np.testing.assert_array_equal(
        payload["cluster_labels"],
        np.array([[0, 3, 2, 1], [3, 0, 1, 2], [0, 1, 3, 2]]),
    )
    np.testing.assert_allclose(
        payload["x_record"],
        np.array(
            [
                [
                    -0.04545209715542766,
                    0.9563681048236234,
                    0.6832212378395283,
                    0.6648412697350994,
                ],
                [
                    0.9263795270955607,
                    -0.3466612849138426,
                    0.27808764055024543,
                    0.31265203806958264,
                ],
                [
                    -0.24185543228176498,
                    0.7876615172785664,
                    0.8474251456092492,
                    0.8189412528994947,
                ],
            ]
        ),
        rtol=1e-9,
    )

    with np.load(output_path) as saved:
        np.testing.assert_array_equal(saved["cluster_labels"], payload["cluster_labels"])
        np.testing.assert_allclose(saved["x_record"], payload["x_record"], rtol=1e-9)


# params: n_sites=4, a_values=[1.4, 1.6], n_transient=5, n_measure=10, seed=7
def test_compute_collective_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "collective_golden.npz"
    payload = compute_collective(
        n_sites=4,
        a_values=np.array([1.4, 1.6]),
        n_transient=5,
        n_measure=10,
        renorm_interval=5,
        seed=7,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(payload["a_values"], np.array([1.4, 1.6]), rtol=1e-9)
    np.testing.assert_allclose(
        payload["lyap_c"],
        np.array([-0.11970016300271111, 0.17056225610801404]),
        rtol=1e-9,
    )
    assert payload["lyap_c"].shape == (2,)
    assert np.all(np.isfinite(payload["lyap_c"]))

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["lyap_c"], payload["lyap_c"], rtol=1e-9)


# params: n_grid=4, n_transient=2, reference_transient=2, period=2, seed=N/A
def test_compute_coupled_basins_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "basins_golden.npz"
    payload = compute_coupled_basins(
        n_grid=4,
        n_transient=2,
        reference_transient=2,
        period=2,
        output_path=output_path,
    )

    np.testing.assert_allclose(
        payload["x"],
        np.array([-1.0, -0.33333333333333337, 0.33333333333333326, 1.0]),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        payload["y"],
        np.array([-1.0, -0.33333333333333337, 0.33333333333333326, 1.0]),
        rtol=1e-9,
    )
    np.testing.assert_array_equal(
        payload["basin"],
        np.array(
            [[0, 1, 1, 1], [2, 0, 1, 1], [2, 2, 0, 1], [2, 2, 2, 0]],
            dtype=np.int8,
        ),
    )

    with np.load(output_path) as saved:
        np.testing.assert_array_equal(saved["basin"], payload["basin"])


# params: A_values=[0.8, 1.0, 1.2], D_values=[0.0, 0.1], n_transient=5, seed=N/A
def test_compute_coupled_phase_diagram_golden_values(tmp_path):
    """Pin CURRENT aggregate output; catches silent numeric drift."""
    output_path = tmp_path / "phase_golden.npz"
    payload = compute_coupled_phase_diagram(
        A_values=np.array([0.8, 1.0, 1.2]),
        D_values=np.array([0.0, 0.1]),
        n_transient=5,
        n_sample=5,
        output_path=output_path,
        progress_interval=0,
    )

    np.testing.assert_allclose(
        payload["asym"],
        np.array(
            [
                [
                    8.2293861882232425e-03,
                    5.9876141248649259e-05,
                    3.0112671401466472e-02,
                ],
                [
                    8.0161532921512221e-02,
                    1.3210878773309020e-02,
                    4.5589216524537955e-03,
                ],
            ]
        ),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        payload["lyap"],
        np.array(
            [
                [-0.0575617156373394, -4.103877162577709, 0.07108327581839047],
                [-0.04912671322212415, -0.5509515979462181, 0.08136671975106122],
            ]
        ),
        rtol=1e-9,
    )

    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["asym"], payload["asym"], rtol=1e-9)
        np.testing.assert_allclose(saved["lyap"], payload["lyap"], rtol=1e-9)
